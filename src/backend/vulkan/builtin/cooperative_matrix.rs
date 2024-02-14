use std::sync::Arc;

use crate::backend::vulkan::vulkan_core::{
    buffer::{Buffer, BufferInfo},
    graph::RGraph,
};
use crate::backend::vulkan::VulkanDevice;

pub fn multiply(
    device: &VulkanDevice,
    regraph: &mut RGraph,
    max_m: usize,
    max_k: usize,
    max_n: usize,
    config: Arc<Buffer>,
    mat_a: Arc<Buffer>,
    mat_b: Arc<Buffer>,
    mat_c: Arc<Buffer>,
) {
    let subgroup_size = device
        .device
        .physical_device
        .subgroup_properties
        .subgroup_size;
}
