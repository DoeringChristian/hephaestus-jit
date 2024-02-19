use std::sync::Arc;

use crate::backend::vulkan::{codegen, VulkanDevice};
use crate::{
    backend::vulkan::vulkan_core::{
        buffer::{Buffer, BufferInfo},
        graph::RGraph,
    },
    vartype::MatMulConfig,
};

use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};

#[derive(Hash)]
enum Activation {
    ReLU,
}

#[derive(Hash)]
struct MLPConfig {
    n_input: u32,
    n_output: u32,
    n_hidden: u32,
    width: u32,
}

#[derive(Hash)]
pub struct MLPCompileDef {
    config: MLPConfig,
}
impl codegen::CodegenDef for MLPCompileDef {
    fn generate(&self) -> Vec<u32> {
        todo!()
    }
}

pub fn mlp_forward(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    config: &MLPConfig,

    input: Arc<Buffer>,
    weights: Arc<Buffer>,
    output: Arc<Buffer>,
) {
    todo!()
}
