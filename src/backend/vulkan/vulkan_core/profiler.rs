use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_profiler::backend::ash::VulkanProfilerFrame;

use super::buffer::{Buffer, BufferInfo};
use super::device::Device;

pub struct ProfilerBuffer(Buffer);

impl gpu_profiler::backend::ash::VulkanBuffer for ProfilerBuffer {
    fn mapped_slice(&self) -> &[u8] {
        self.0.mapped_slice()
    }

    fn raw(&self) -> ash::vk::Buffer {
        self.0.vk()
    }
}

pub struct ProfilerBackend {
    device: Device,
}

impl ProfilerBackend {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl gpu_profiler::backend::ash::VulkanBackend for ProfilerBackend {
    type Buffer = ProfilerBuffer;

    fn create_query_result_buffer(&mut self, bytes: usize) -> Self::Buffer {
        ProfilerBuffer(Buffer::create(
            &self.device,
            BufferInfo {
                size: bytes,
                alignment: 0,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuToCpu,
            },
        ))
    }

    fn timestamp_period(&self) -> f32 {
        self.device
            .physical_device
            .properties
            .limits
            .timestamp_period
    }
}

pub type ProfilerData = VulkanProfilerFrame<ProfilerBuffer>;
