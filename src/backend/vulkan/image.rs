use std::fmt::{Debug, Formatter};
use std::ops::Deref;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
pub use gpu_allocator::MemoryLocation;

use super::device::Device;
use super::VulkanDevice;

pub struct Image {
    allocation: Option<Allocation>,
    device: Device,
    image: vk::Image,
    info: ImageInfo,
}

impl Image {
    pub fn create(device: &Device, info: &ImageInfo) -> Self {
        let create_info = vk::ImageCreateInfo::builder()
            .image_type(info.ty)
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(vk::Extent3D {
                width: info.width,
                height: info.height,
                depth: info.depth,
            })
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = device
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap()
        };

        Self {
            allocation: Some(allocation),
            device: device.clone(),
            image,
            info: *info,
        }
    }
}

impl Deref for Image {
    type Target = vk::Image;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl Debug for Image {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.image)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            unsafe {
                self.device.destroy_image(self.image, None);
            }

            self.device
                .allocator
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free(allocation)
                .unwrap();
        }
    }
}

impl Image {
    pub fn from_buffer(&self, cb: vk::CommandBuffer, device: &VulkanDevice) {
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C)]
        struct Copy2D {
            width: u32,
            height: u32,
            src_pitch: u32,
            dst_pitch: u32,
        }
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageInfo {
    pub ty: vk::ImageType,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}
