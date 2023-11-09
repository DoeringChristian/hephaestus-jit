use std::ffi::CStr;

use crate::backend::vulkan::codegen;
use crate::ir::IR;

use super::buffer::Buffer;
use super::device::Device;
use ash::vk;

pub struct Pipeline {
    device: Device,
    desc_sets: Vec<vk::DescriptorSet>,
    desc_set_layouts: Vec<vk::DescriptorSetLayout>,
    desc_pool: vk::DescriptorPool,
    // descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            for desc_set_layout in self.desc_set_layouts.iter() {
                self.device
                    .destroy_descriptor_set_layout(*desc_set_layout, None);
            }
        }
    }
}

impl Pipeline {
    pub fn from_ir(device: &Device, ir: &IR) -> Self {
        let spirv = codegen::assemble_trace(ir, "main").unwrap();

        let num_buffers = ir.n_buffers;

        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::builder()
                .code(spirv.as_slice())
                .build();
            let shader = device.create_shader_module(&shader_info, None).unwrap();

            // Create Descriptor Pool
            let desc_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: num_buffers as _,
            }];
            let desc_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&desc_sizes)
                .max_sets(1);
            let desc_pool = device
                .create_descriptor_pool(&desc_pool_info, None)
                .unwrap();

            // Create Layout
            let desc_layout_bindings = [vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: num_buffers as _,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            }];
            let desc_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

            let desc_set_layouts = vec![device
                .create_descriptor_set_layout(&desc_info, None)
                .unwrap()];

            // Allocate Descriptor Sets
            let desc_sets_allocation_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(desc_pool)
                .set_layouts(&desc_set_layouts);
            let desc_sets = device
                .allocate_descriptor_sets(&desc_sets_allocation_info)
                .unwrap();

            // Create Pipeline
            let pipeline_layout_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layouts);
            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap();
            let pipeline_cache = device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder(), None)
                .unwrap();

            let pipeline_shader_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(CStr::from_bytes_with_nul(b"main\0").unwrap());

            let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .stage(pipeline_shader_info.build())
                .layout(pipeline_layout);
            let compute_pipeline = device
                .create_compute_pipelines(pipeline_cache, &[compute_pipeline_info.build()], None)
                .unwrap()[0];

            // Destruct temporary elements
            device.destroy_shader_module(shader, None);
            device.destroy_pipeline_cache(pipeline_cache, None);

            Self {
                device: device.clone(),
                desc_sets,
                pipeline_layout,
                pipeline: compute_pipeline,
                desc_set_layouts,
                desc_pool,
            }
        }
    }
    pub fn launch_fenced<'a>(&'a self, num: usize, buffers: impl Iterator<Item = &'a Buffer>) {
        let desc_buffer_infos = buffers
            .map(|buffer| vk::DescriptorBufferInfo {
                buffer: buffer.buffer(),
                offset: 0,
                range: buffer.info().size as _,
            })
            .collect::<Vec<_>>();
        let write_desc_sets = [vk::WriteDescriptorSet::builder()
            .dst_set(self.desc_sets[0])
            .buffer_info(&desc_buffer_infos)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_binding(0)
            .build()];

        unsafe {
            self.device.update_descriptor_sets(&write_desc_sets, &[]);

            self.device.submit_global(|device, cb| {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &self.desc_sets,
                    &[],
                );
                device.cmd_dispatch(cb, num as _, 1, 1);
            });
        }
    }
}
