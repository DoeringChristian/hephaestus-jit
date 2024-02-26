mod accel;
mod builtin;
mod codegen;
mod pipeline;
mod shader_cache;
#[cfg(test)]
mod test;
mod utils;
mod vkdevice;
mod vulkan_core;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::backend;
use crate::backend::vulkan::builtin::{cooperative_matrix, fused_mlp};
use crate::backend::vulkan::vulkan_core::graph::RGraph;
use crate::ir::IR;
use crate::op::DeviceOp;
use crate::vartype::{AsVarType, FusedMlpConfig};
use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;
use vulkan_core::buffer::{Buffer, BufferInfo};
use vulkan_core::device::Device;
use vulkan_core::image::{Image, ImageInfo};

use self::codegen::DeviceInfo;
use self::pipeline::PipelineDesc;
use self::shader_cache::{ShaderCache, ShaderKind};

/// TODO: Find better way to chache pipelines
#[derive(Debug)]
pub struct InternalVkDevice {
    device: Device,
    pipeline_cache: Mutex<HashMap<u64, Arc<pipeline::Pipeline>>>,
    shader_cache: Mutex<ShaderCache>,
}
impl InternalVkDevice {
    fn compile_ir(&self, ir: &IR, info: &DeviceInfo) -> Arc<pipeline::Pipeline> {
        let mut hasher = DefaultHasher::new();
        ir.internal_hash().hash(&mut hasher);
        info.hash(&mut hasher);
        let hash = hasher.finish();

        self.pipeline_cache
            .lock()
            .unwrap()
            .entry(hash)
            .or_insert_with(|| Arc::new(pipeline::Pipeline::from_ir(&self.device, ir, info)))
            .clone()
    }
    fn get_pipeline<'a>(&'a self, desc: &PipelineDesc<'a>) -> Arc<pipeline::Pipeline> {
        self.pipeline_cache
            .lock()
            .unwrap()
            .entry(desc.hash())
            .or_insert_with(|| Arc::new(pipeline::Pipeline::create(&self.device, desc)))
            .clone()
    }
    fn get_shader<I: codegen::CodegenDef + 'static>(&self, input: &I) -> Arc<Vec<u32>> {
        self.shader_cache.lock().unwrap().get(input)
    }
    fn get_shader_glsl(
        &self,
        src: &str,
        kind: ShaderKind,
        defines: &[(&str, Option<&str>)],
    ) -> Arc<Vec<u32>> {
        self.shader_cache
            .lock()
            .unwrap()
            .lease_glsl(src, kind, defines)
    }
}
impl std::ops::Deref for InternalVkDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

#[derive(Clone, Debug)]
pub struct VulkanDevice(Arc<InternalVkDevice>);
impl VulkanDevice {
    #[profiling::function]
    pub fn create(id: usize) -> backend::Result<Self> {
        Ok(Self(Arc::new(InternalVkDevice {
            device: Device::create(id),
            pipeline_cache: Default::default(),
            shader_cache: Default::default(),
        })))
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = InternalVkDevice;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl backend::BackendDevice for VulkanDevice {
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;
    type Accel = VulkanAccel;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        // WARN: compress and prefix_sum rely on the buffer being divisible by 16
        // Therefore we allocate with powers of 2
        let size = crate::utils::u64::round_pow2(size as _);
        let info = BufferInfo {
            size: size as usize,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            memory_location: MemoryLocation::GpuOnly,
        };
        let buffer = Arc::new(Buffer::create(self, info));
        Ok(VulkanBuffer {
            buffer,
            device: self.clone(),
        })
    }

    #[profiling::function]
    fn execute_graph(
        &self,
        graph: &crate::graph::Graph,
        env: &crate::graph::Env,
    ) -> backend::Result<backend::Report> {
        use crate::graph::PassOp;
        let mut rgraph = RGraph::new();

        for (i, pass) in graph.passes.iter().enumerate() {
            let to_buffer = |id: crate::graph::ResourceId| {
                env.buffer(id)
                    .and_then(|buffer| Some(buffer.vulkan()?.buffer.clone()))
            };
            let to_image = |id: crate::graph::ResourceId| {
                env.texture(id)
                    .and_then(|image| Some(image.vulkan()?.image.clone()))
            };
            let to_accel = |id: crate::graph::ResourceId| {
                env.accel(id)
                    .and_then(|accel| Some(accel.vulkan()?.accel.clone()))
            };
            let buffers = pass
                .resources
                .iter()
                .flat_map(|id| env.buffer(*id))
                .map(|buffer| buffer.vulkan().unwrap().buffer.clone())
                .collect::<Vec<_>>();
            let images = pass
                .resources
                .iter()
                .flat_map(|id| env.texture(*id))
                .map(|texture| texture.vulkan().unwrap().image.clone())
                .collect::<Vec<_>>();
            let accels = pass
                .resources
                .iter()
                .flat_map(|id| env.accel(*id))
                .map(|accel| accel.vulkan().unwrap().accel.clone())
                .collect::<Vec<_>>();
            match &pass.op {
                PassOp::Kernel { ir, size } => {
                    let size = *size;
                    let compile_info = DeviceInfo {
                        work_group_size: 128,
                    };
                    let pipeline = self.compile_ir(ir, &compile_info);
                    // TODO: if we ever add dynamic sized kernels pass the buffer here

                    // If the pass contains a size_buffer, then use it otherwise create one with
                    // the static size.
                    let size_buffer = pass
                        .size_buffer
                        .and_then(|id| env.buffer(id))
                        .map(|buffer| buffer.vulkan().unwrap().buffer.clone())
                        .unwrap_or_else(|| {
                            let mut size_buffer = Buffer::create(
                                self,
                                BufferInfo {
                                    size: std::mem::size_of::<u32>(),
                                    usage: vk::BufferUsageFlags::TRANSFER_SRC
                                        | vk::BufferUsageFlags::TRANSFER_DST
                                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                                    memory_location: MemoryLocation::CpuToGpu,
                                    ..Default::default()
                                },
                            );
                            size_buffer
                                .mapped_slice_mut()
                                .copy_from_slice(bytemuck::cast_slice(&[size as u32]));
                            Arc::new(size_buffer)
                        });

                    let buffers = [size_buffer]
                        .into_iter()
                        .chain(buffers.into_iter())
                        .collect::<Vec<_>>();

                    // Create a render pass on the graph, pushing all it's resource accesses
                    let mut rpass = rgraph.pass(format!("JIT Kernel {i}"));
                    for buffer in &buffers {
                        rpass = rpass.read(&buffer, AccessType::ComputeShaderReadOther);
                        rpass = rpass.write(&buffer, AccessType::ComputeShaderWrite);
                    }
                    for image in &images {
                        rpass = rpass.read(
                            &image,
                            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                        );
                    }
                    for accel in &accels {
                        rpass = rpass.read(&accel.tlas, AccessType::ComputeShaderReadOther);
                        for blas in accel.blases.iter() {
                            rpass = rpass.read(&blas, AccessType::ComputeShaderReadOther);
                        }
                    }

                    let grid_size = (size + compile_info.work_group_size as usize - 1)
                        / compile_info.work_group_size as usize;

                    rpass.record(move |device, cb, pool| {
                        pipeline.submit_to_cbuffer(cb, pool, grid_size, &buffers, &images, &accels);
                    });
                }
                PassOp::DeviceOp(op) => match op {
                    DeviceOp::ReduceOp(op) => {
                        let dst = buffers[0].clone();
                        let src = buffers[1].clone();
                        let ty = &graph.buffer_desc(pass.resources[0]).ty;
                        let num = graph.buffer_desc(pass.resources[1]).size;
                        builtin::reduce::reduce(&self, &mut rgraph, *op, ty, num, &src, &dst);
                    }
                    DeviceOp::PrefixSum { inclusive } => {
                        let dst = buffers[0].clone();
                        let src = buffers[1].clone();
                        let ty = &graph.buffer_desc(pass.resources[0]).ty;
                        let num = graph.buffer_desc(pass.resources[1]).size;
                        builtin::prefix_sum::prefix_sum(
                            &self,
                            &mut rgraph,
                            ty,
                            num,
                            *inclusive,
                            &src,
                            &dst,
                        );
                    }
                    DeviceOp::Compress => {
                        let index_out = to_buffer(pass.resources[0]).unwrap();
                        let out_count = to_buffer(pass.resources[1]).unwrap();
                        let src = to_buffer(pass.resources[2]).unwrap();

                        let size_buffer = pass.size_buffer.and_then(to_buffer);

                        let num = graph.buffer_desc(pass.resources[2]).size;

                        builtin::compress::compress(
                            &self,
                            &mut rgraph,
                            num,
                            size_buffer,
                            &out_count,
                            &src,
                            &index_out,
                        );
                    }
                    DeviceOp::MatMul {
                        max_n,
                        max_m,
                        max_k,
                    } => {
                        let mat_d = to_buffer(pass.resources[0]).unwrap();
                        let mat_a = to_buffer(pass.resources[1]).unwrap();
                        let mat_b = to_buffer(pass.resources[2]).unwrap();
                        let mat_c = to_buffer(pass.resources[3]).unwrap();
                        let config = pass.resources.get(4).cloned().and_then(to_buffer);

                        let a_type = &graph.buffer_desc(pass.resources[1]).ty;
                        let c_type = &graph.buffer_desc(pass.resources[3]).ty;

                        cooperative_matrix::multiply(
                            &self,
                            &mut rgraph,
                            a_type,
                            c_type,
                            *max_n as _,
                            *max_m as _,
                            *max_k as _,
                            config,
                            mat_a,
                            mat_b,
                            mat_c,
                            mat_d,
                        );
                    }
                    DeviceOp::FusedMlpInference {
                        width,
                        in_width,
                        out_width,
                        hidden_layers,
                        max_batch_size,
                    } => {
                        let output = to_buffer(pass.resources[0]).unwrap();
                        let input = to_buffer(pass.resources[1]).unwrap();
                        let weights = to_buffer(pass.resources[2]).unwrap();
                        let config = pass.resources.get(3).cloned().and_then(to_buffer);
                        fused_mlp::mlp_inference(
                            &self,
                            &mut rgraph,
                            input,
                            weights,
                            output,
                            config,
                            FusedMlpConfig {
                                batch_size: *max_batch_size as _,
                            },
                            *width,
                            *in_width,
                            *out_width,
                            *hidden_layers,
                        );
                    }
                    DeviceOp::Buffer2Texture => {
                        let src = buffers[0].clone();
                        let dst = images[0].clone();
                        dst.copy_from_buffer(&mut rgraph, &src);
                    }
                    DeviceOp::BuildAccel => {
                        let accel_desc = graph.accel_desc(pass.resources[0]);
                        self.build_accel(&mut rgraph, &accel_desc, &accels[0], buffers.iter());
                    }
                },
                _ => todo!(),
            }
        }
        let execution_report = rgraph.submit(self);

        Ok(backend::Report {
            exec: execution_report,
        })
    }

    fn create_texture(&self, desc: &backend::TextureDesc) -> backend::Result<Self::Texture> {
        let dim = desc.shape.iter().take_while(|d| **d > 0).count();
        assert!(
            dim >= 1 && dim <= 3,
            "{dim} dimensional textures are not supported.
                Only 1, 2 and 3 dimensional textures are supported!",
            dim = desc.shape.len()
        );
        assert!(desc.channels <= 4);

        let width = desc.shape[0].max(1) as _;
        let height = desc.shape[1].max(1) as _;
        let depth = desc.shape[2].max(1) as _;
        let ty = match dim {
            1 => vk::ImageType::TYPE_1D,
            2 => vk::ImageType::TYPE_2D,
            3 => vk::ImageType::TYPE_3D,
            _ => todo!(),
        };
        let format = utils::channels_ty_to_format(desc.channels, desc.format);

        let image = Arc::new(Image::create(
            self,
            ImageInfo {
                ty,
                format,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth,
                },
            },
        ));

        Ok(Self::Texture {
            image,
            // device: self.clone(),
            // shape: Vec::from(shape),
            // channels,
        })
    }

    fn create_buffer_from_slice(&self, slice: &[u8]) -> backend::Result<Self::Buffer> {
        let size = slice.len();
        let buffer = self.create_buffer(size)?;

        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            memory_location: MemoryLocation::CpuToGpu,
        };
        let mut staging = Buffer::create(&self.device, info);
        staging.mapped_slice_mut().copy_from_slice(slice);
        self.device.submit_global(|device, cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: size as _,
            };
            device.cmd_copy_buffer(cb, staging.vk(), buffer.buffer.vk(), &[region]);
        });
        Ok(buffer)
    }

    fn create_accel(&self, desc: &backend::AccelDesc) -> backend::Result<Self::Accel> {
        Ok(VulkanAccel {
            accel: Arc::new(accel::Accel::create(&self, desc)),
        })
    }
}

#[derive(Clone)]
pub struct VulkanBuffer {
    buffer: Arc<Buffer>,
    device: VulkanDevice,
}

impl Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("size", &self.buffer.info().size)
            .finish()
    }
}

impl backend::BackendBuffer for VulkanBuffer {
    type Device = VulkanDevice;

    fn to_host<T: AsVarType + Copy>(
        &self,
        range: std::ops::Range<usize>,
    ) -> backend::Result<Vec<T>> {
        log::trace!("Copying {buffer:?} to host", buffer = self.buffer);
        let len = range.len();
        let ty_size = T::var_ty().size();
        let size = len * ty_size;

        assert!(self.buffer.size() >= size);

        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuToCpu,
        };
        let staging = Buffer::create(&self.device, info);
        self.device.submit_global(|device, cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: (range.start * ty_size) as _,
                dst_offset: 0,
                size: size as _,
            };
            device.cmd_copy_buffer(cb, self.buffer.vk(), staging.vk(), &[region]);
        });
        Ok(
            unsafe { std::slice::from_raw_parts(staging.mapped_slice().as_ptr() as *const T, len) }
                .to_vec(),
        )
        // Ok(bytemuck::cast_slice(staging.mapped_slice()).to_vec())
    }

    // fn size(&self) -> usize {
    //     self.buffer.info().size
    // }
    fn device(&self) -> &Self::Device {
        &self.device
    }
}

#[derive(Debug, Clone)]
pub struct VulkanTexture {
    image: Arc<Image>,
    // device: VulkanDevice,
    // shape: Vec<usize>,
    // channels: usize,
}

impl backend::BackendTexture for VulkanTexture {
    type Device = VulkanDevice;
}

impl VulkanTexture {
    fn copy_from_buffer(&self, rgraph: &mut RGraph, src: &Arc<Buffer>) {
        // assert!(self.channels <= 4);
        self.image.copy_from_buffer(rgraph, src);
    }
}

#[derive(Debug, Clone)]
pub struct VulkanAccel {
    accel: Arc<accel::Accel>,
}

impl backend::BackendAccel for VulkanAccel {
    type Device = VulkanDevice;
}
