use std::collections::HashSet;
use std::ffi::CStr;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use ash;
use ash::vk;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{0}")]
    VkResult(#[from] vk::Result),
    #[error("Could not find a Queue Family, supporting the required features!")]
    QueueFamilyNotFound,
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct PhysicalDevice {
    pub physical_device: vk::PhysicalDevice,

    pub features2: vk::PhysicalDeviceFeatures2<'static>,
    pub features_v1_1: vk::PhysicalDeviceVulkan11Features<'static>,
    pub features_v1_2: vk::PhysicalDeviceVulkan12Features<'static>,
    pub acceleration_structure_features:
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    pub ray_query_features: vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    pub cooperative_matrix_features: vk::PhysicalDeviceCooperativeMatrixFeaturesKHR<'static>,

    pub properties: vk::PhysicalDeviceProperties,

    pub properties2: vk::PhysicalDeviceProperties2<'static>,
    pub properties_v1_1: vk::PhysicalDeviceVulkan11Properties<'static>,
    pub properties_v1_2: vk::PhysicalDeviceVulkan12Properties<'static>,
    pub acceleration_structure_properties:
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    pub subgroup_properties: vk::PhysicalDeviceSubgroupProperties<'static>,
    pub cooperative_matrix_properties: vk::PhysicalDeviceCooperativeMatrixPropertiesKHR<'static>,

    pub extensions: Vec<vk::ExtensionProperties>,

    pub queue_family_index: u32,

    pub memory_properties: vk::PhysicalDeviceMemoryProperties,

    pub supports_accel_struct: bool,
    pub supports_index_type_uint8: bool,
    pub supports_ray_query: bool,
    pub supports_ray_trace: bool,
}

impl Debug for PhysicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            write!(
                f,
                "{name:?}",
                name = CStr::from_ptr(self.properties.device_name.as_ptr())
            )
        }
    }
}

impl PhysicalDevice {
    pub fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Result<Self> {
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

        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let mut properties_v1_1 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut properties_v1_2 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut acceleration_structure_properties =
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut cooperative_matrix_properties =
            vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();

        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut properties_v1_1)
            .push_next(&mut properties_v1_2)
            .push_next(&mut acceleration_structure_properties)
            .push_next(&mut subgroup_properties)
            .push_next(&mut cooperative_matrix_properties);

        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };

        let extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };

        unsafe {
            log::trace!(
                "Found Physical Device {physical_device:?} {name:?}:",
                name = CStr::from_ptr(properties.device_name.as_ptr())
            );
            log::trace!("Extensions:");
            for extension in extensions.iter() {
                log::trace!(
                    "    {name:?}: {version} ",
                    name = CStr::from_ptr(extension.extension_name.as_ptr()),
                    version = extension.spec_version
                );
            }
            log::trace!("Features:");
        }

        let extension_names = extensions
            .iter()
            .map(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
            .collect::<HashSet<_>>();

        let supports_accel_struct = extension_names.contains(vk::KhrAccelerationStructureFn::NAME)
            && extension_names.contains(vk::KhrDeferredHostOperationsFn::NAME);
        let supports_index_type_uint8 = extension_names.contains(vk::ExtIndexTypeUint8Fn::NAME);
        let supports_ray_query = extension_names.contains(vk::KhrRayQueryFn::NAME);
        let supports_ray_trace = extension_names.contains(vk::KhrRayTracingPipelineFn::NAME);

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find_map(|(index, info)| {
                    let valid = info.queue_flags.contains(vk::QueueFlags::COMPUTE);
                    if valid {
                        Some(index as u32)
                    } else {
                        None
                    }
                })
                .ok_or(Error::QueueFamilyNotFound)?
        };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        unsafe {
            Ok(Self {
                physical_device,
                features2: std::mem::transmute(features2),
                features_v1_1: std::mem::transmute(features_v1_1),
                features_v1_2: std::mem::transmute(features_v1_2),
                acceleration_structure_features,
                ray_query_features,
                cooperative_matrix_features: std::mem::transmute(cooperative_matrix_features),

                properties,
                properties2: std::mem::transmute(properties2),
                properties_v1_1: std::mem::transmute(properties_v1_1),
                properties_v1_2: std::mem::transmute(properties_v1_2),
                acceleration_structure_properties: std::mem::transmute(
                    acceleration_structure_properties,
                ),
                subgroup_properties: std::mem::transmute(subgroup_properties),
                cooperative_matrix_properties: std::mem::transmute(cooperative_matrix_properties),

                extensions,
                queue_family_index,
                memory_properties,

                supports_ray_query,
                supports_ray_trace,
                supports_accel_struct,
                supports_index_type_uint8,
            })
        }
    }
}
