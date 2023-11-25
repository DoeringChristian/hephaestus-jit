mod accel;
mod acceleration_structure;
mod buffer;
mod codegen;
mod context;
mod device;
mod glslext;
mod image;
mod physical_device;
mod pipeline;
#[cfg(test)]
mod test;

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use crate::backend;
use crate::ir::IR;
use crate::op::DeviceOp;
use ash::vk;
use buffer::{Buffer, BufferInfo};
use device::Device;
use gpu_allocator::MemoryLocation;
use image::{Image, ImageInfo};

use self::context::Context;
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

impl backend::BackendDevice for VulkanDevice {
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;
    type Accel = VulkanAccel;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
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

        todo!();
        // pipeline.launch_fenced(num, buffers.iter().map(|b| &b.buffer));

        Ok(())
    }

    fn execute_graph(
        &self,
        trace: &crate::tr::Trace,
        graph: &crate::graph::Graph,
    ) -> backend::Result<()> {
        use crate::graph::PassOp;
        // WARN: Potential Use after Free (GPU) when references are droped before cbuffer has ben
        // submitted
        // FIX: Add a struct that can collect Arcs to those resources
        self.submit_global(|ctx| {
            for pass in graph.passes.iter() {
                let buffers = pass
                    .buffers
                    .iter()
                    .map(|id| {
                        let buffer = graph.buffer(trace, *id);
                        &buffer.vulkan().unwrap().buffer
                    })
                    .collect::<Vec<_>>();
                let images = pass
                    .textures
                    .iter()
                    .map(|id| {
                        let buffer = graph.texture(trace, *id);
                        &buffer.vulkan().unwrap().image
                    })
                    .collect::<Vec<_>>();
                let accels = pass
                    .accels
                    .iter()
                    .map(|id| {
                        let accel = graph.accel(trace, *id);
                        &accel.vulkan().unwrap().accel
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
                            ctx.cmd_pipeline_barrier(
                                ctx.cb,
                                vk::PipelineStageFlags::ALL_COMMANDS,
                                vk::PipelineStageFlags::ALL_COMMANDS,
                                vk::DependencyFlags::empty(),
                                &memory_barriers,
                                &[],
                                &[],
                            );
                        }
                        pipeline.submit_to_cbuffer(ctx, *size, &buffers, &images, &accels);
                    }
                    PassOp::DeviceOp(op) => match op {
                        DeviceOp::Max => todo!(),
                        DeviceOp::Buffer2Texture { .. } => {
                            let src = graph.buffer(trace, pass.buffers[0]).vulkan().unwrap();
                            let dst = graph.texture(trace, pass.textures[0]).vulkan().unwrap();
                            dst.copy_from_buffer(ctx, &src.buffer);
                        }
                        DeviceOp::BuildAccel => {
                            let accel_desc = graph.accel_desc(pass.accels[0]);
                            let accel = trace
                                .var(accel_desc.var.id())
                                .data
                                .accel()
                                .unwrap()
                                .vulkan()
                                .unwrap();

                            // WARN: This is potentially very unsafe, since we are just randomly
                            // accessing the buffers and hoping for them to be index/vertex
                            // buffers
                            let mut buffers =
                                pass.buffers.iter().map(|id| graph.buffer(trace, *id));

                            let instances = &buffers.next().unwrap().vulkan().unwrap().buffer;

                            let geometries = accel_desc
                                .desc
                                .geometries
                                .iter()
                                .map(|g| match g {
                                    backend::GeometryDesc::Triangles { .. } => {
                                        accel::AccelGeometryBuildInfo::Triangles {
                                            triangles: &buffers
                                                .next()
                                                .unwrap()
                                                .vulkan()
                                                .unwrap()
                                                .buffer,
                                            vertices: &buffers
                                                .next()
                                                .unwrap()
                                                .vulkan()
                                                .unwrap()
                                                .buffer,
                                        }
                                    }
                                })
                                .collect::<Vec<_>>();

                            let desc = accel::AccelBuildInfo {
                                geometries: &geometries,
                                instances,
                            };

                            accel.accel.build(ctx, desc);
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
        assert!(channels <= 4);

        let width = shape[0].max(1) as _;
        let height = shape[1].max(1) as _;
        let depth = shape[2].max(1) as _;
        let ty = match dim {
            1 => vk::ImageType::TYPE_1D,
            2 => vk::ImageType::TYPE_2D,
            3 => vk::ImageType::TYPE_3D,
            _ => todo!(),
        };
        let format = match channels {
            1 => vk::Format::R32_SFLOAT,
            2 => vk::Format::R32G32_SFLOAT,
            3 => vk::Format::R32G32B32_SFLOAT,
            4 => vk::Format::R32G32B32A32_SFLOAT,
            _ => todo!(),
        };

        let image = Image::create(
            self,
            &ImageInfo {
                ty,
                format,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth,
                },
            },
        );

        Ok(Self::Texture {
            image,
            device: self.clone(),
            shape: Vec::from(shape),
            channels,
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
        self.device.submit_global(|ctx| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: size as _,
            };
            ctx.cmd_copy_buffer(ctx.cb, staging.buffer(), buffer.buffer.buffer(), &[region]);
        });
        Ok(buffer)
    }

    fn create_accel(&self, desc: &backend::AccelDesc) -> backend::Result<Self::Accel> {
        Ok(VulkanAccel {
            accel: accel::Accel::create(&self, desc),
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
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuToCpu,
        };
        let staging = Buffer::create(&self.device, info);
        self.device.submit_global(|ctx| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: self.size() as _,
            };
            ctx.cmd_copy_buffer(ctx.cb, self.buffer.buffer(), staging.buffer(), &[region]);
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
    image: Image,
    device: VulkanDevice,
    shape: Vec<usize>,
    channels: usize,
}

impl backend::BackendTexture for VulkanTexture {
    type Device = VulkanDevice;
}

impl VulkanTexture {
    fn copy_from_buffer(&self, ctx: &Context, src: &Buffer) {
        assert!(self.channels <= 4);
        self.image.copy_from_buffer(ctx, src);
    }
}

#[derive(Debug)]
pub struct VulkanAccel {
    accel: accel::Accel,
}

impl backend::BackendAccel for VulkanAccel {
    type Device = VulkanDevice;
}
