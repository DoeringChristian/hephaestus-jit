use std::ffi::CStr;

use super::device::Device;
use ash::extensions::khr;
use ash::vk;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

#[derive(Debug)]
pub struct Presenter {
    device: Device,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
    present_mode: vk::PresentModeKHR,
    swapchain_create_info: vk::SwapchainCreateInfoKHR,
    swapchain: vk::SwapchainKHR,
    present_images: Vec<vk::Image>,
    present_image_views: Vec<vk::ImageView>,
}

impl Drop for Presenter {
    fn drop(&mut self) {
        unsafe {
            self.device
                .surface_ext
                .as_ref()
                .unwrap()
                .destroy_surface(self.surface, None);
            for image_view in self.present_image_views.iter() {
                self.device.destroy_image_view(*image_view, None);
            }
            for image in self.present_images.iter() {
                self.device.destroy_image(*image, None);
            }
            self.device
                .swapchain_ext
                .as_ref()
                .unwrap()
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

pub struct PresenterInfo {
    width: u32,
    height: u32,
}

impl Presenter {
    pub fn create(device: &Device, info: &PresenterInfo) -> Self {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Hephaesus")
            .with_inner_size(winit::dpi::LogicalSize::new(
                f64::from(info.width),
                f64::from(info.height),
            ))
            .build(&event_loop)
            .unwrap();

        let required_extensions =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .into_iter()
                .map(|s| unsafe { CStr::from_ptr(*s) })
                .collect::<Vec<_>>();
        log::trace!("Presenting Images requires the following extensions (for now just hope they are present):");
        log::trace!("{required_extensions:#?}");

        let surface = unsafe {
            ash_window::create_surface(
                &device.entry,
                &device.instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .unwrap()
        };

        let surface_format = unsafe {
            device
                .surface_ext
                .as_ref()
                .unwrap()
                .get_physical_device_surface_formats(
                    device.physical_device.physical_device,
                    surface,
                )
                .unwrap()[0]
        };
        let surface_capabilities = unsafe {
            device
                .surface_ext
                .as_ref()
                .unwrap()
                .get_physical_device_surface_capabilities(
                    device.physical_device.physical_device,
                    surface,
                )
                .unwrap()
        };

        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: info.width,
                height: info.height,
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let present_modes = unsafe {
            device
                .surface_ext
                .as_ref()
                .unwrap()
                .get_physical_device_surface_present_modes(
                    device.physical_device.physical_device,
                    surface,
                )
                .unwrap()
        };
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .build();

        let swapchain = unsafe {
            device
                .swapchain_ext
                .as_ref()
                .unwrap()
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };

        let present_images = unsafe {
            device
                .swapchain_ext
                .as_ref()
                .unwrap()
                .get_swapchain_images(swapchain)
                .unwrap()
        };

        let present_image_views: Vec<vk::ImageView> = present_images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe { device.create_image_view(&create_view_info, None).unwrap() }
            })
            .collect();

        Self {
            device: device.clone(),
            surface,
            surface_format,
            surface_capabilities,
            present_mode,
            swapchain_create_info,
            swapchain,
            present_images,
            present_image_views,
        }
    }
}
