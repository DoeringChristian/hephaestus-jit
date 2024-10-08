use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt::Debug;
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex};

use ash::ext::debug_utils;
// use ash::extensions::ext::DebugUtils;
use ash::Entry;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings};

pub use ash::{ext, khr, vk};
use itertools::Itertools;

use super::physical_device::PhysicalDevice;
use super::{buffer, image, pipeline, pool, Error, Result};

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::trace!("{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n"),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n"),
        _ => {}
    };

    vk::FALSE
}

pub struct Device {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub debug_utils_loader: debug_utils::Instance,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
    pub physical_device: PhysicalDevice,

    pub queue: vk::Queue,
    pub pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    pub allocator: Option<Mutex<Allocator>>,

    pub fence: Mutex<vk::Fence>,

    pub acceleration_structure_ext: Option<khr::acceleration_structure::Device>,
    pub cooperative_matrix_ext: Option<khr::cooperative_matrix::Instance>,
    pub cooperative_matrix_properties: Vec<vk::CooperativeMatrixPropertiesKHR<'static>>,

    pub buffer_pool: pool::ResourcePool<buffer::InternalBuffer>,
    pub image_pool: pool::ResourcePool<image::InternalImage>,
    pub pipeline_cache: Mutex<HashMap<u64, Arc<pipeline::InternalPipeline>>>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}
impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InternalDevice")
            .field("physical_device", &self.physical_device)
            .field("queue", &self.queue)
            .field("pool", &self.pool)
            .field("command_buffer", &self.command_buffer)
            .field("allocator", &self.allocator)
            .field("fence", &self.fence)
            .finish()
    }
}
impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Device {
    pub fn create(index: usize) -> Result<Arc<Self>> {
        unsafe {
            let entry = Entry::linked();
            let app_name = CStr::from_bytes_with_nul_unchecked(b"Candle\0");

            let layer_names =
                [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];
            // let layer_names = [];

            let extension_names = [debug_utils::NAME.as_ptr()];

            let appinfo = vk::ApplicationInfo::default()
                .application_name(app_name)
                .application_version(0)
                .engine_name(app_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 2, 0));

            let create_flags = vk::InstanceCreateFlags::default();

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&appinfo)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&extension_names)
                .flags(create_flags);

            let instance = entry.create_instance(&create_info, None)?;
            // .map_err(|_| VulkanError::InstanceCreationError)?;

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
            let debug_callback =
                debug_utils_loader.create_debug_utils_messenger(&debug_info, None)?;

            let mut physical_devices = instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|physical_device| Ok(PhysicalDevice::new(&instance, physical_device)?))
                .collect::<Result<Vec<_>>>()?;
            physical_devices.sort_by_key(|physical_device| {
                match physical_device.properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 3,
                    vk::PhysicalDeviceType::OTHER => 4,
                    _ => todo!(),
                }
            });

            log::trace!("Compatible Devices: {physical_devices:?}");
            let physical_device = physical_devices
                .into_iter()
                .nth(index)
                .ok_or(Error::PhysicalDeviceNotFound(index))?;

            let mut device_extension_names = vec![ext::descriptor_indexing::NAME.as_ptr()];

            if physical_device.supports_ray_query {
                device_extension_names.push(khr::ray_query::NAME.as_ptr());
            }
            if physical_device.supports_accel_struct {
                device_extension_names.push(khr::acceleration_structure::NAME.as_ptr());
                device_extension_names.push(khr::deferred_host_operations::NAME.as_ptr());
            }
            if physical_device.supports_cooperative_matrix {
                device_extension_names.push(khr::cooperative_matrix::NAME.as_ptr());
            }

            let queue_family_index = physical_device.queue_family_index;
            let vk_physical_device = physical_device.physical_device;

            // TODO: better features for physical device
            let mut features_v1_1 = vk::PhysicalDeviceVulkan11Features::default();
            let mut features_v1_2 = vk::PhysicalDeviceVulkan12Features::default();
            let mut acceleration_structure_features =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let mut cooperative_matrix_features =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features_v1_1)
                .push_next(&mut features_v1_2)
                .push_next(&mut acceleration_structure_features)
                .push_next(&mut ray_query_features)
                .push_next(&mut cooperative_matrix_features);

            instance.get_physical_device_features2(physical_device.physical_device, &mut features2);

            let priorities = [1.0];

            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names)
                .push_next(&mut features2);

            let device = instance.create_device(vk_physical_device, &device_create_info, None)?;
            // .map_err(|_| VulkanError::DeviceCreateError)?;

            let queue = device.get_device_queue(physical_device.queue_family_index, 0);

            // let device_memory_properties =
            //     instance.get_physical_device_memory_properties(physical_device.physical_device);

            let pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(physical_device.queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffer = device.allocate_command_buffers(&command_buffer_allocate_info)?[0];

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: physical_device.physical_device,
                debug_settings: AllocatorDebugSettings {
                    log_leaks_on_shutdown: true,
                    log_memory_information: true,
                    log_allocations: true,
                    ..Default::default()
                },
                buffer_device_address: true,
                allocation_sizes: AllocationSizes::new(u64::MAX, u64::MAX),
            })?;

            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence = device.create_fence(&fence_info, None)?;

            // Extensons:
            let acceleration_structure_ext = physical_device
                .supports_accel_struct
                .then(|| khr::acceleration_structure::Device::new(&instance, &device));

            let cooperative_matrix_ext = physical_device
                .supports_cooperative_matrix
                .then(|| khr::cooperative_matrix::Instance::new(&entry, &instance));

            let cooperative_matrix_properties = cooperative_matrix_ext
                .as_ref()
                .into_iter()
                .map(|ext| {
                    Ok(ext.get_physical_device_cooperative_matrix_properties(
                        physical_device.physical_device,
                    )?)
                })
                .flatten_ok()
                .collect::<Result<Vec<_>>>()?;
            // Cast away livetime constraints
            let cooperative_matrix_properties = std::mem::transmute(cooperative_matrix_properties);
            log::trace!("The following cooperative matrices are supported: {cooperative_matrix_properties:#?}");

            Ok(Arc::new(Self {
                entry,
                instance,
                device,
                debug_utils_loader,
                debug_callback,
                physical_device,
                queue,
                pool,
                allocator: Some(Mutex::new(allocator)),
                command_buffer,
                fence: Mutex::new(fence),
                acceleration_structure_ext,
                cooperative_matrix_ext,
                cooperative_matrix_properties,

                buffer_pool: Default::default(),
                image_pool: Default::default(),
                pipeline_cache: Mutex::new(Default::default()),
            }))
        }
    }

    #[profiling::function]
    pub fn submit_global<'a, F: FnOnce(&Self, vk::CommandBuffer)>(
        &'a self,
        f: F,
    ) -> (std::time::SystemTime, std::time::Duration) {
        let fence = self.fence.lock().unwrap();
        let result = unsafe {
            // Record command buffer
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.begin_command_buffer(self.command_buffer, &command_buffer_begin_info)
                .unwrap();
            f(self, self.command_buffer);
            self.end_command_buffer(self.command_buffer).unwrap();

            // Wait for fences and submit command buffer
            self.reset_fences(&[*fence]).unwrap();

            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            let start_system;
            let start;
            {
                profiling::scope!("Submit and Wait");
                start_system = std::time::SystemTime::now();
                start = std::time::Instant::now();
                self.queue_submit(self.queue, &[submit_info], *fence)
                    .unwrap();

                self.wait_for_fences(&[*fence], true, u64::MAX).unwrap();
            }
            let end = std::time::Instant::now();

            (start_system, end - start)
        };
        drop(fence);
        result
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Clear pools:
            self.buffer_pool.clear(self);
            self.image_pool.clear(self);
            for pipeline in self.pipeline_cache.lock().unwrap().values_mut() {
                Arc::get_mut(pipeline).unwrap().destroy(self);
            }

            self.device_wait_idle().unwrap();

            self.allocator.take().unwrap();
            self.destroy_command_pool(self.pool, None);
            self.destroy_fence(*self.fence.lock().unwrap(), None);
            self.destroy_device(None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}
