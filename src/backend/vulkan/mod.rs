mod buffer;
mod codegen;
mod device;
mod glslext;
mod pipeline;

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use crate::backend;
use crate::ir::IR;
use ash::vk;
use buffer::{Buffer, BufferInfo};
use device::Device;
use gpu_allocator::MemoryLocation;

/// TODO: Find better way to chache pipelines
#[derive(Debug)]
pub struct InternalVkDevice {
    device: Device,
    pipeline_cache: Mutex<HashMap<u64, Arc<pipeline::Pipeline>>>,
}
impl InternalVkDevice {
    fn get_pipeline(&self, ir: &IR) -> Arc<pipeline::Pipeline> {
        self.pipeline_cache
            .lock()
            .unwrap()
            .entry(ir.hash())
            .or_insert_with(|| Arc::new(pipeline::Pipeline::from_ir(&self.device, ir)))
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
        let pipeline = self.get_pipeline(ir);

        pipeline.launch_fenced(num, buffers.iter().map(|b| &b.buffer));

        Ok(())
    }

    fn execute_graph(
        &self,
        trace: &crate::tr::Trace,
        graph: &crate::graph::Graph,
    ) -> backend::Result<()> {
        use crate::graph::Op;
        self.submit_global(|device, cb| {
            for pass in graph.passes.iter() {
                let buffers = pass
                    .buffers
                    .iter()
                    .map(|id| {
                        let desc = graph.buffer_desc(*id);
                        let buffer = trace.var(desc.var.id()).data.buffer().unwrap().clone();
                        match buffer {
                            backend::Buffer::VulkanBuffer(buffer) => buffer,
                            _ => todo!(),
                        }
                    })
                    .collect::<Vec<_>>();
                match &pass.op {
                    Op::Kernel { ir, size } => {
                        let pipeline = self.get_pipeline(ir);

                        // TODO: look at barriers
                        let barrier = vk::MemoryBarrier2KHR {
                            src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER_KHR,
                            src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER_KHR,
                            dst_access_mask: vk::AccessFlags2::SHADER_WRITE,
                            ..Default::default()
                        };
                        let info = vk::DependencyInfoKHR::builder()
                            .memory_barriers(&[barrier])
                            .build();
                        unsafe {
                            device.cmd_pipeline_barrier2(cb, &info);
                        }
                        pipeline.submit_to_cbuffer(
                            cb,
                            device,
                            *size,
                            buffers.iter().map(|b| &b.buffer),
                        );
                    }
                    _ => todo!(),
                }
            }
        });
        Ok(())
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
