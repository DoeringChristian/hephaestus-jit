use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
pub use gpu_allocator::MemoryLocation;

use crate::utils;

use super::device::Device;

#[derive(Debug)]
pub struct Buffer {
    allocation: Option<Allocation>,
    buffer: vk::Buffer,
    device: Device,

    info: BufferInfo,
}
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferInfo {
    pub size: usize,
    pub alignment: usize,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: MemoryLocation,
}
impl Default for BufferInfo {
    fn default() -> Self {
        Self {
            size: Default::default(),
            alignment: Default::default(),
            usage: Default::default(),
            memory_location: MemoryLocation::GpuOnly,
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        self.device
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .free(self.allocation.take().unwrap())
            .unwrap();

        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

impl Buffer {
    pub fn device(&self) -> &Device {
        &self.device
    }
    #[profiling::function]
    pub fn create(device: &Device, info: BufferInfo) -> Self {
        let device = device.clone();
        let queue_family_indices = [device.physical_device.queue_family_index];
        let buffer_info = vk::BufferCreateInfo::default()
            .size(info.size as _)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices);
        let buffer = unsafe {
            device.create_buffer(&buffer_info, None).unwrap()
            // .map_err(|_| VulkanError::CreatteBufferError)?
        };
        let mut requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        requirements.alignment = requirements.alignment.max(info.alignment as _);

        let allocation = device
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: info.memory_location,
                linear: true, // Buffers are always linear
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            device
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        }
        log::trace!("Created Buffer with id {buffer:?}.");

        Self {
            allocation: Some(allocation),
            buffer,
            device,
            info,
        }
    }
    pub fn mapped_slice(&self) -> &[u8] {
        &self.allocation.as_ref().unwrap().mapped_slice().unwrap()[0..self.info.size as usize]
    }
    pub fn mapped_slice_mut(&mut self) -> &mut [u8] {
        &mut self
            .allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .unwrap()[0..self.info.size as usize]
    }
    pub fn info(&self) -> BufferInfo {
        self.info
    }
    pub fn size(&self) -> usize {
        self.info().size
    }
    pub fn vk(&self) -> vk::Buffer {
        self.buffer
    }
    pub fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.buffer),
            )
        }
    }
    pub fn create_mapped_storage(device: &Device, data: &[u8]) -> Self {
        let mut buffer = Self::create(
            device,
            BufferInfo {
                size: data.len(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
        );
        buffer.mapped_slice_mut().copy_from_slice(data);

        buffer
    }
}
