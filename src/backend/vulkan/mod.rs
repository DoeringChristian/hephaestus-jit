mod buffer;
mod codegen;
mod device;
mod param_layout;

use crate::backend;
use ash::vk::{self, CommandPoolCreateInfo};
use buffer::{Buffer, BufferInfo};
use device::Device;
use gpu_allocator::MemoryLocation;
use param_layout::ParamLayout;

#[derive(Clone, Debug)]
pub struct VulkanDevice(Device);
impl VulkanDevice {
    pub fn create(id: usize) -> backend::Result<Self> {
        Ok(Self(Device::create(0)))
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl backend::BackendDevice for VulkanDevice {
    type Array = VulkanArray;

    fn create_array(&self, size: usize) -> backend::Result<Self::Array> {
        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        };
        let buffer = Buffer::create(self, info);
        Ok(VulkanArray {
            buffer,
            device: self.clone(),
        })
    }

    fn execute_trace(
        &self,
        trace: &crate::trace::Trace,
        params: backend::Parameters,
    ) -> backend::Result<()> {
        let layout = ParamLayout::generate(trace);
        codegen::assemble_trace(trace, "main");
        todo!()
    }
}

#[derive(Debug)]
pub struct VulkanArray {
    buffer: Buffer,
    device: VulkanDevice,
}

impl backend::BackendArray for VulkanArray {
    type Device = VulkanDevice;

    fn to_host(&self) -> backend::Result<Vec<u8>> {
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
        Ok(staging.mapped_slice().to_vec())
    }

    fn size(&self) -> usize {
        self.buffer.info().size
    }
}
