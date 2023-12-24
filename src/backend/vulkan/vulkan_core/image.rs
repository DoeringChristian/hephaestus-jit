use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
pub use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use super::buffer::{Buffer, BufferInfo};
use super::device::Device;
use super::graph::{Access, RGraph};

pub struct Image {
    allocation: Option<Allocation>,
    device: Device,
    image: vk::Image,
    sampler: vk::Sampler,
    info: ImageInfo,
    image_view_cache: Mutex<HashMap<ImageViewInfo, vk::ImageView>>,
}

impl Image {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn create(device: &Device, info: ImageInfo) -> Self {
        log::trace!("Creating Image with {info:?}");
        let create_info = vk::ImageCreateInfo {
            image_type: info.ty,
            format: info.format,
            extent: info.extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        log::trace!("Creating VkImage with {create_info:#?}");

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

        // Create Default sampler
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .unnormalized_coordinates(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .build();
        let sampler = unsafe { device.create_sampler(&sampler_info, None).unwrap() };

        Self {
            allocation: Some(allocation),
            device: device.clone(),
            image,
            sampler,
            image_view_cache: Mutex::new(Default::default()),
            info,
        }
    }
    pub fn copy_from_buffer(self: &Arc<Self>, rgraph: &mut RGraph, src: &Arc<Buffer>) {
        let region = vk::BufferImageCopy::builder()
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .build(),
            )
            .image_extent(self.info().extent)
            .build();

        let s = self.clone();

        {
            let s = self.clone();
            let src = src.clone();
            rgraph
                .pass()
                .read(&src, AccessType::TransferRead)
                .write(&s, AccessType::TransferWrite)
                .record(move |device, cb, _| unsafe {
                    device.cmd_copy_buffer_to_image(
                        cb,
                        src.vk(),
                        s.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    );
                });
        }
    }
    pub fn default_sampler(&self) -> vk::Sampler {
        self.sampler
    }
    pub fn vk(&self) -> vk::Image {
        self.image
    }
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }
    pub fn image_view(&self, info: ImageViewInfo) -> vk::ImageView {
        *self
            .image_view_cache
            .lock()
            .unwrap()
            .entry(info)
            .or_insert_with_key(|info| {
                let mut info: vk::ImageViewCreateInfo = (*info).into();
                info.image = self.image;
                unsafe { self.device.create_image_view(&info, None).unwrap() }
            })
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
                self.device.destroy_sampler(self.sampler, None);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageInfo {
    pub ty: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageViewInfo {
    pub format: vk::Format,
    pub ty: vk::ImageViewType,

    // SubresourceRange:
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl From<ImageViewInfo> for vk::ImageViewCreateInfo {
    fn from(value: ImageViewInfo) -> Self {
        vk::ImageViewCreateInfo {
            flags: vk::ImageViewCreateFlags::empty(),
            view_type: value.ty,
            format: value.format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: value.aspect_mask,
                base_mip_level: value.base_mip_level,
                level_count: value.level_count,
                base_array_layer: value.base_array_layer,
                layer_count: value.layer_count,
            },
            ..Default::default()
        }
    }
}
