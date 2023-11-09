mod buffer;
mod codegen;
mod device;
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

#[derive(Debug)]
pub struct InternalDevice {
    device: Device,
    pipeline_cache: Mutex<HashMap<u64, Arc<pipeline::Pipeline>>>,
}
impl InternalDevice {
    fn get_pipeline(&self, ir: &mut IR) -> Arc<pipeline::Pipeline> {
        self.pipeline_cache
            .lock()
            .unwrap()
            .entry(ir.hash())
            .or_insert_with(|| Arc::new(pipeline::Pipeline::from_ir(&self.device, ir)))
            .clone()
    }
}
impl std::ops::Deref for InternalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

#[derive(Clone, Debug)]
pub struct VulkanDevice(Arc<InternalDevice>);
impl VulkanDevice {
    pub fn create(id: usize) -> backend::Result<Self> {
        Ok(Self(Arc::new(InternalDevice {
            device: Device::create(id),
            pipeline_cache: Default::default(),
        })))
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = InternalDevice;

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
        let pipeline = pipeline::Pipeline::from_ir(&self, ir);

        pipeline.launch_fenced(num, buffers.iter().map(|b| &b.buffer));

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
