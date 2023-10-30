use std::borrow::Cow;
use std::ffi::CStr;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use ash::extensions::ext::DebugUtils;
use ash::Entry;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::AllocatorDebugSettings;

pub use ash::vk;

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

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}

#[derive(Debug, Clone)]
pub struct Device(Arc<InternalDevice>);
impl Device {
    pub fn create(index: usize) -> Self {
        Self(Arc::new(InternalDevice::create(index)))
    }
    pub fn submit_global<F: FnOnce(&Self, vk::CommandBuffer)>(&self, f: F) {
        unsafe {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.begin_command_buffer(self.command_buffer, &command_buffer_begin_info)
                .unwrap();

            f(self, self.command_buffer);

            self.end_command_buffer(self.command_buffer).unwrap();

            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&[self.command_buffer])
                .build();

            self.queue_submit(self.queue, &[submit_info], self.fence)
                .unwrap();

            self.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            self.reset_fences(&[self.fence]).unwrap();
        }
    }
}
impl Deref for Device {
    type Target = Arc<InternalDevice>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct InternalDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub debug_utils_loader: DebugUtils,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,

    pub queue: vk::Queue,
    pub pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    pub allocator: Option<Mutex<Allocator>>,

    pub fence: vk::Fence,
}
impl Debug for InternalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InternalDevice")
            .field("physical_device", &self.physical_device)
            .field("device_memory_properties", &self.device_memory_properties)
            .field("queue_family_index", &self.queue_family_index)
            .field("queue", &self.queue)
            .field("pool", &self.pool)
            .field("command_buffer", &self.command_buffer)
            .field("allocator", &self.allocator)
            .field("fence", &self.fence)
            .finish()
    }
}

impl InternalDevice {
    pub fn create(index: usize) -> Self {
        unsafe {
            let entry = Entry::linked();
            let app_name = CStr::from_bytes_with_nul_unchecked(b"Candle\0");

            let layer_names =
                [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];

            let extension_names = [DebugUtils::name().as_ptr()];

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(app_name)
                .application_version(0)
                .engine_name(app_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 3, 0));

            let create_flags = vk::InstanceCreateFlags::default();

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&extension_names)
                .flags(create_flags);

            let instance = entry.create_instance(&create_info, None).unwrap();
            // .map_err(|_| VulkanError::InstanceCreationError)?;

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
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

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_callback = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let physical_devices = instance.enumerate_physical_devices().unwrap();
            let (physical_device, queue_family_index) = physical_devices
                .iter()
                .filter_map(|physical_device| {
                    instance
                        .get_physical_device_queue_family_properties(*physical_device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let valid = info.queue_flags.contains(vk::QueueFlags::COMPUTE);
                            if valid {
                                Some((*physical_device, index as _))
                            } else {
                                None
                            }
                        })
                })
                .nth(index)
                .unwrap();

            // let queue_family_index =
            // instance.get_physical_device_queue_family_properties(pdevice)

            let device_extension_names = [];

            let features = vk::PhysicalDeviceFeatures {
                ..Default::default()
            };

            let priorities = [1.0];

            let queue_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names)
                .enabled_features(&features);

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap();
            // .map_err(|_| VulkanError::DeviceCreateError)?;

            let queue = device.get_device_queue(queue_family_index, 0);

            let device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffer = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()[0];

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: AllocatorDebugSettings {
                    log_leaks_on_shutdown: true,
                    log_memory_information: true,
                    log_allocations: true,
                    ..Default::default()
                },
                buffer_device_address: false,
            })
            .unwrap();

            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let fence = device.create_fence(&fence_info, None).unwrap();

            Self {
                entry,
                instance,
                device,
                debug_utils_loader,
                debug_callback,
                physical_device,
                device_memory_properties,
                queue_family_index,
                queue,
                pool,
                allocator: Some(Mutex::new(allocator)),
                command_buffer,
                fence,
            }
        }
    }
}

impl Deref for InternalDevice {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}
impl Drop for InternalDevice {
    fn drop(&mut self) {
        unsafe {
            self.device_wait_idle().unwrap();

            self.allocator.take().unwrap();
            self.destroy_command_pool(self.pool, None);
            self.destroy_fence(self.fence, None);
            self.destroy_device(None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}
