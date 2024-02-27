use super::device::Device;
use ash::vk;

#[derive(Debug)]
pub struct Pipeline {
    desc_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Pipeline {}

pub struct PipelineRef {
    device: Device,
}
