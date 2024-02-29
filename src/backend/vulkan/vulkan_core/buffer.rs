use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
pub use gpu_allocator::MemoryLocation;

use crate::utils;

use super::device::{self, Device};
use super::pool::{Lease, Resource};

pub struct Buffer {
    device: Arc<Device>,
    lease: Lease<InternalBuffer>,
    info: BufferInfo,
}
impl Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("info", &self.info).finish()
    }
}

#[derive(Debug)]
pub(super) struct InternalBuffer {
    allocation: Option<Allocation>,
    buffer: vk::Buffer,
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

impl Resource for InternalBuffer {
    type Info = BufferInfo;

    fn create(device: &Device, info: &Self::Info) -> Self {
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
        }
    }
    fn destroy(&mut self, device: &device::Device) {
        device
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .free(self.allocation.take().unwrap())
            .unwrap();

        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

impl Device {
    pub fn create_buffer_type<T: Sized>(self: &Arc<Self>, info: BufferInfo) -> Buffer {
        let lease_info = BufferInfo {
            size: utils::usize::align_up(info.size, 2),
            alignment: info.alignment.max(std::mem::align_of::<T>()),
            usage: info.usage,
            memory_location: info.memory_location,
        };
        let lease = self.buffer_pool.lease(self, &lease_info);
        Buffer {
            device: self.clone(),
            lease,
            info,
        }
    }
}

impl Buffer {
    // pub fn device(&self) -> &Device {
    //     &self.device
    // }
    #[profiling::function]
    pub fn create(device: &Arc<Device>, info: BufferInfo) -> Self {
        let lease_info = BufferInfo {
            size: utils::usize::align_up(info.size, 2),
            alignment: info.alignment,
            usage: info.usage,
            memory_location: info.memory_location,
        };
        let lease = device.buffer_pool.lease(device, &lease_info);
        Self {
            device: device.clone(),
            lease,
            info,
        }
    }
    pub fn mapped_slice(&self) -> &[u8] {
        &self
            .lease
            .allocation
            .as_ref()
            .unwrap()
            .mapped_slice()
            .unwrap()[0..self.info.size as usize]
    }
    pub fn mapped_slice_mut(&mut self) -> &mut [u8] {
        &mut self
            .lease
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
        self.lease.buffer
    }
    pub fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.lease.buffer),
            )
        }
    }

    pub fn create_mapped_storage(device: &Arc<Device>, data: &[u8]) -> Self {
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
