use std::fmt::{Debug, Formatter};
use std::ops::Deref;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
pub use gpu_allocator::MemoryLocation;

use super::buffer::{Buffer, BufferInfo};
use super::device::Device;
use super::VulkanDevice;

pub struct Image {
    allocation: Option<Allocation>,
    device: Device,
    image: vk::Image,
    sampler: vk::Sampler,
    info: ImageInfo,
}

impl Image {
    pub fn create(device: &Device, info: &ImageInfo) -> Self {
        log::trace!("Creating Image with {info:?}");
        let queue_family_indices = [device.queue_family_index];
        let create_info = vk::ImageCreateInfo::builder()
            .image_type(info.ty)
            .format(info.format)
            .extent(vk::Extent3D {
                width: info.width,
                height: info.height,
                depth: info.depth,
            })
            .mip_levels(1)
            .array_layers(1)
            .queue_family_indices(&queue_family_indices)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .build();
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
            info: *info,
        }
    }
    pub fn copy_from_buffer(&self, cb: vk::CommandBuffer, device: &Device, src: &Buffer) {
        // let memory_barriers = [vk::MemoryBarrier::builder()
        //     .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        //     .dst_access_mask(vk::AccessFlags::SHADER_READ)
        //     .build()];
        let image_memory_barreirs = [vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(self.image)
            .build()];
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barreirs,
            );
        }
        let region = vk::BufferImageCopy::builder()
            .image_extent(vk::Extent3D {
                width: self.info().width,
                height: self.info().height,
                depth: self.info().depth,
            })
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        unsafe {
            device.cmd_copy_buffer_to_image(
                cb,
                src.buffer(),
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // let memory_barriers = [vk::MemoryBarrier::builder()
        //     .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        //     .dst_access_mask(vk::AccessFlags::SHADER_READ)
        //     .build()];
        let image_memory_barreirs = [vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(self.image)
            .build()];
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barreirs,
            );
        }
    }
    pub fn default_sampler(&self) -> vk::Sampler {
        self.sampler
    }
    pub fn n_texels(&self) -> usize {
        self.info.width as usize * self.info.height as usize * self.info.depth as usize
    }
    pub fn info(&self) -> &ImageInfo {
        &self.info
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

#[derive(Clone, Copy, Debug)]
pub struct ImageInfo {
    pub ty: vk::ImageType,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}
