mod buffer;
mod codegen;
mod device;
mod glslext;
mod image;
mod pipeline;
#[cfg(test)]
mod test;

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use crate::backend;
use crate::backend::vulkan::pipeline::{Binding, BufferWriteInfo, DescSetLayout, WriteSet};
use crate::ir::IR;
use crate::op::DeviceOp;
use ash::vk;
use buffer::{Buffer, BufferInfo};
use device::Device;
use gpu_allocator::MemoryLocation;
use image::{Image, ImageInfo};

use self::pipeline::PipelineDesc;

/// TODO: Find better way to chache pipelines
#[derive(Debug)]
pub struct InternalVkDevice {
    device: Device,
    pipeline_cache: Mutex<HashMap<u64, Arc<pipeline::Pipeline>>>,
}
impl InternalVkDevice {
    fn compile_ir(&self, ir: &IR) -> Arc<pipeline::Pipeline> {
        self.pipeline_cache
            .lock()
            .unwrap()
            .entry(ir.hash())
            .or_insert_with(|| Arc::new(pipeline::Pipeline::from_ir(&self.device, ir)))
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
    pub fn create(id: usize) -> backend::Result<Self> {
        Ok(Self(Arc::new(InternalVkDevice {
            device: Device::create(id),
            pipeline_cache: Default::default(),
        })))
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = InternalVkDevice;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl VulkanDevice {
    fn device_op(&self, cb: vk::CommandBuffer, op: DeviceOp) {
        match op {
            DeviceOp::Max => todo!(),
            DeviceOp::Buffer2Texture { .. } => todo!(),
        }
    }
}

impl backend::BackendDevice for VulkanDevice {
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        };
        let buffer = Buffer::create(self, info);
        Ok(VulkanBuffer {
            buffer,
            device: self.clone(),
        })
    }

    fn execute_ir(&self, ir: &IR, num: usize, buffers: &[&Self::Buffer]) -> backend::Result<()> {
        let pipeline = self.compile_ir(ir);

        pipeline.launch_fenced(num, buffers.iter().map(|b| &b.buffer));

        Ok(())
    }

    fn execute_graph(
        &self,
        trace: &crate::tr::Trace,
        graph: &crate::graph::Graph,
    ) -> backend::Result<()> {
        use crate::graph::PassOp;
        self.submit_global(|device, cb| {
            for pass in graph.passes.iter() {
                let buffers = pass
                    .buffers
                    .iter()
                    .map(|id| {
                        let buffer = graph.buffer(trace, *id);
                        match buffer {
                            backend::Buffer::VulkanBuffer(buffer) => buffer,
                            _ => todo!(),
                        }
                    })
                    .collect::<Vec<_>>();
                match &pass.op {
                    PassOp::Kernel { ir, size } => {
                        let pipeline = self.compile_ir(ir);

                        let memory_barriers = [vk::MemoryBarrier::builder()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)
                            .build()];
                        unsafe {
                            device.cmd_pipeline_barrier(
                                cb,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &memory_barriers,
                                &[],
                                &[],
                            );
                        }
                        pipeline.submit_to_cbuffer(
                            cb,
                            device,
                            *size,
                            buffers.iter().map(|b| &b.buffer),
                        );
                    }
                    PassOp::DeviceOp(op) => match op {
                        DeviceOp::Max => todo!(),
                        DeviceOp::Buffer2Texture { .. } => {
                            let src = graph.buffer(trace, pass.buffers[0]).vulkan().unwrap();
                            let dst = graph.texture(trace, pass.textures[0]).vulkan().unwrap();
                            dst.copy_from_buffer(cb, self, &src.buffer);
                        }
                    },
                    _ => todo!(),
                }
            }
        });
        Ok(())
    }

    fn create_texture(&self, shape: [usize; 3], channels: usize) -> backend::Result<Self::Texture> {
        let dim = shape.iter().take_while(|d| **d > 0).count();
        assert!(
            dim >= 1 && dim <= 3,
            "{dim} dimensional textures are not supported.
                Only 1, 2 and 3 dimensional textures are supported!",
            dim = shape.len()
        );

        let images = (0..channels)
            .step_by(4)
            .map(|i| {
                let width = shape[0].min(1) as _;
                let height = shape[1].min(1) as _;
                let depth = shape[2].min(1) as _;

                let ty = match dim {
                    1 => vk::ImageType::TYPE_1D,
                    2 => vk::ImageType::TYPE_2D,
                    3 => vk::ImageType::TYPE_3D,
                    _ => todo!(),
                };

                let image = Image::create(
                    self,
                    &ImageInfo {
                        ty,
                        width,
                        height,
                        depth,
                    },
                );

                image
            })
            .collect::<Vec<_>>();
        Ok(Self::Texture {
            images,
            device: self.clone(),
            shape: Vec::from(shape),
            channels,
        })
    }
}

pub struct VulkanBuffer {
    buffer: Buffer,
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

    fn to_host<T: bytemuck::Pod>(&self) -> backend::Result<Vec<T>> {
        let size = self.size();
        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuToCpu,
        };
        let staging = Buffer::create(&self.device, info);
        self.device.submit_global(|device, cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: self.size() as _,
            };
            device.cmd_copy_buffer(cb, self.buffer.buffer(), staging.buffer(), &[region]);
        });
        Ok(bytemuck::cast_slice(staging.mapped_slice()).to_vec())
    }

    fn size(&self) -> usize {
        self.buffer.info().size
    }
    fn device(&self) -> &Self::Device {
        &self.device
    }
}

#[derive(Debug)]
pub struct VulkanTexture {
    images: Vec<Image>,
    device: VulkanDevice,
    shape: Vec<usize>,
    channels: usize,
}

impl backend::BackendTexture for VulkanTexture {
    type Device = VulkanDevice;
}

impl VulkanTexture {
    fn copy_from_buffer(&self, cb: vk::CommandBuffer, device: &VulkanDevice, src: &Buffer) {
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C)]
        struct Copy2D {
            width: u32,
            height: u32,
            src_pitch: u32,
            dst_pitch: u32,
            src_offset: u32,
            dst_offset: u32,
        }
        let channels_of_image = |i: usize| ((i + 1) * 4).min(self.channels - i * 4);

        let internal_channels = 4;

        let staging_buffer = Buffer::create(
            &device,
            BufferInfo {
                size: self.images[0].n_texels() * internal_channels * std::mem::size_of::<f32>(),
                alignment: 0,
                usage: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        for (i, image) in self.images.iter().enumerate() {
            let cfg = Copy2D {
                width: channels_of_image(i) as _,
                height: image.n_texels() as _,
                src_pitch: self.channels as _,
                dst_pitch: internal_channels as _,
                src_offset: (i * internal_channels) as _,
                dst_offset: 0,
            };
            let size = cfg.width * cfg.height;
            let mut cfg_buffer = Buffer::create(
                device,
                BufferInfo {
                    size: std::mem::size_of::<Copy2D>(),
                    alignment: 0,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    memory_location: MemoryLocation::CpuToGpu,
                },
            );
            cfg_buffer
                .mapped_slice_mut()
                .copy_from_slice(bytemuck::cast_slice(&[cfg]));

            let pipeline = device.get_pipeline(&PipelineDesc {
                code: inline_spirv::include_spirv!("src/backend/vulkan/kernels/copy2d.glsl", comp),
                desc_set_layouts: &[DescSetLayout {
                    bindings: &[
                        Binding {
                            binding: 0,
                            count: 1,
                        },
                        Binding {
                            binding: 1,
                            count: 1,
                        },
                        Binding {
                            binding: 2,
                            count: 1,
                        },
                    ],
                }],
            });
            pipeline.submit(
                cb,
                &device,
                &[
                    WriteSet {
                        set: 0,
                        binding: 0,
                        buffers: &[BufferWriteInfo {
                            buffer: &cfg_buffer,
                        }],
                    },
                    WriteSet {
                        set: 0,
                        binding: 1,
                        buffers: &[BufferWriteInfo { buffer: &src }],
                    },
                    WriteSet {
                        set: 0,
                        binding: 2,
                        buffers: &[BufferWriteInfo {
                            buffer: &staging_buffer,
                        }],
                    },
                ],
                (size, 1, 1),
            );

            let memory_barriers = [vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build()];
            let image_memory_barreirs = [vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(**image)
                .build()];
            unsafe {
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &memory_barriers,
                    &[],
                    &image_memory_barreirs,
                );
            }
            let region = vk::BufferImageCopy::builder()
                .image_extent(vk::Extent3D {
                    width: image.info().width,
                    height: image.info().height,
                    depth: image.info().depth,
                })
                .build();
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cb,
                    staging_buffer.buffer(),
                    **image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }

            let memory_barriers = [vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build()];
            let image_memory_barreirs = [vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(**image)
                .build()];
            unsafe {
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &memory_barriers,
                    &[],
                    &image_memory_barreirs,
                );
            }
        }
        todo!();
    }
}
