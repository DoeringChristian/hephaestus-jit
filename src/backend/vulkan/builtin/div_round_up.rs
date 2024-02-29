use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use crate::backend::vulkan::builtin::utils::GlslShaderDef;

use crate::backend::vulkan::{codegen, VulkanDevice};
use crate::utils::usize;
use crate::{
    backend::vulkan::vulkan_core::{
        buffer::{Buffer, BufferInfo},
        graph::RGraph,
    },
    vartype::FusedMlpConfig,
};

use crate::backend::vulkan::vulkan_core::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineInfo, WriteSet,
};

pub fn div_round_up(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    output: Arc<Buffer>,
    value: Arc<Buffer>,
    divisor: Arc<Buffer>,
) {
}
