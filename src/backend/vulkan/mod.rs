mod buffer;
mod codegen;
mod device;
mod param_layout;

use std::ffi::CStr;

use crate::backend;
use ash::vk::{self, CommandPoolCreateInfo};
use buffer::{Buffer, BufferInfo};
use device::Device;
use gpu_allocator::MemoryLocation;
use param_layout::ParamLayout;

#[derive(Clone, Debug)]
pub struct VulkanDevice(Device);
impl VulkanDevice {
    pub fn create(id: usize) -> backend::Result<Self> {
        Ok(Self(Device::create(0)))
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl backend::BackendDevice for VulkanDevice {
    type Array = VulkanArray;

    fn create_array(&self, size: usize) -> backend::Result<Self::Array> {
        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        };
        let buffer = Buffer::create(self, info);
        Ok(VulkanArray {
            buffer,
            device: self.clone(),
        })
    }

    fn execute_trace(
        &self,
        trace: &crate::trace::Trace,
        params: backend::Parameters,
    ) -> backend::Result<()> {
        let spirv = codegen::assemble_trace(trace, "main").unwrap();

        //         let spirv = inline_spirv::inline_spirv!(
        //             r#"
        // #version 460
        //
        // layout(set = 0, binding=0) buffer Test{
        //     uint i[];
        // }test;
        //
        // void main()
        // {
        //         test.i[0] = 0;
        // }
        //         "#,
        //             glsl,
        //             comp
        //         );

        let num = trace.size;

        let num_buffers = params.arrays.len(); // TODO: get from trace

        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::builder()
                .code(spirv.as_slice())
                .build();
            let shader = self.create_shader_module(&shader_info, None).unwrap();

            // Create Descriptor Pool
            let desc_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: num_buffers as _,
            }];
            let desc_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&desc_sizes)
                .max_sets(1);
            let desc_pool = self.create_descriptor_pool(&desc_pool_info, None).unwrap();

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

            let desc_set_layouts = [self.create_descriptor_set_layout(&desc_info, None).unwrap()];

            // Allocate Descriptor Sets
            let desc_sets_allocation_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(desc_pool)
                .set_layouts(&desc_set_layouts);
            let desc_sets = self
                .allocate_descriptor_sets(&desc_sets_allocation_info)
                .unwrap();

            // Create Pipeline
            let pipeline_layout_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layouts);
            let pipeline_layout = self
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap();
            let pipeline_cache = self
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder(), None)
                .unwrap();

            let pipeline_shader_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(CStr::from_bytes_with_nul(b"main\0").unwrap());

            let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .stage(pipeline_shader_info.build())
                .layout(pipeline_layout);
            let compute_pipeline = self
                .create_compute_pipelines(pipeline_cache, &[compute_pipeline_info.build()], None)
                .unwrap()[0];

            let desc_buffer_infos = params
                .arrays
                .iter()
                .map(|array| vk::DescriptorBufferInfo {
                    buffer: array.as_vulkan().unwrap().buffer.buffer(),
                    offset: 0,
                    range: array.size() as _,
                })
                .collect::<Vec<_>>();
            let write_desc_sets = [vk::WriteDescriptorSet::builder()
                .dst_set(desc_sets[0])
                .buffer_info(&desc_buffer_infos)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .dst_binding(0)
                .build()];

            self.update_descriptor_sets(&write_desc_sets, &[]);

            self.submit_global(|device, cb| {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, compute_pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    &desc_sets,
                    &[],
                );
                device.cmd_dispatch(cb, num as _, 1, 1);
            });
        }

        // WARN: Very leaky function
        // TODO: Fix

        Ok(())
    }
}

#[derive(Debug)]
pub struct VulkanArray {
    buffer: Buffer,
    device: VulkanDevice,
}

impl backend::BackendArray for VulkanArray {
    type Device = VulkanDevice;

    fn to_host(&self) -> backend::Result<Vec<u8>> {
        let size = self.size();
        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuToCpu,
        };
        let staging = Buffer::create(&self.device, info);
        self.device.submit_global(|device, cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: self.size() as _,
            };
            device.cmd_copy_buffer(cb, self.buffer.buffer(), staging.buffer(), &[region]);
        });
        Ok(staging.mapped_slice().to_vec())
    }

    fn size(&self) -> usize {
        self.buffer.info().size
    }
}
