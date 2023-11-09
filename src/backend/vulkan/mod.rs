mod buffer;
mod codegen;
mod device;
mod pipeline;

use std::fmt::Debug;

use crate::backend;
use crate::ir::IR;
use ash::vk;
use buffer::{Buffer, BufferInfo};
use device::Device;
use gpu_allocator::MemoryLocation;

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
    type Buffer = VulkanBuffer;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        let info = BufferInfo {
            size,
            alignment: 0,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        };
        let buffer = Buffer::create(self, info);
        Ok(VulkanBuffer {
            buffer,
            device: self.clone(),
        })
    }

    fn execute_ir(&self, ir: &IR, num: usize, buffers: &[&Self::Buffer]) -> backend::Result<()> {
        let pipeline = pipeline::Pipeline::from_ir(&self, ir);

        pipeline.launch_fenced(num, buffers.iter().map(|b| &b.buffer));

        // let num = num;
        //
        // let num_buffers = buffers.len(); // TODO: get from trace
        //
        // unsafe {
        //     let shader_info = vk::ShaderModuleCreateInfo::builder()
        //         .code(spirv.as_slice())
        //         .build();
        //     let shader = self.create_shader_module(&shader_info, None).unwrap();
        //
        //     // Create Descriptor Pool
        //     let desc_sizes = [vk::DescriptorPoolSize {
        //         ty: vk::DescriptorType::STORAGE_BUFFER,
        //         descriptor_count: num_buffers as _,
        //     }];
        //     let desc_pool_info = vk::DescriptorPoolCreateInfo::builder()
        //         .pool_sizes(&desc_sizes)
        //         .max_sets(1);
        //     let desc_pool = self.create_descriptor_pool(&desc_pool_info, None).unwrap();
        //
        //     // Create Layout
        //     let desc_layout_bindings = [vk::DescriptorSetLayoutBinding {
        //         binding: 0,
        //         descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        //         descriptor_count: num_buffers as _,
        //         stage_flags: vk::ShaderStageFlags::ALL,
        //         ..Default::default()
        //     }];
        //     let desc_info =
        //         vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);
        //
        //     let desc_set_layouts = [self.create_descriptor_set_layout(&desc_info, None).unwrap()];
        //
        //     // Allocate Descriptor Sets
        //     let desc_sets_allocation_info = vk::DescriptorSetAllocateInfo::builder()
        //         .descriptor_pool(desc_pool)
        //         .set_layouts(&desc_set_layouts);
        //     let desc_sets = self
        //         .allocate_descriptor_sets(&desc_sets_allocation_info)
        //         .unwrap();
        //
        //     // Create Pipeline
        //     let pipeline_layout_info =
        //         vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layouts);
        //     let pipeline_layout = self
        //         .create_pipeline_layout(&pipeline_layout_info, None)
        //         .unwrap();
        //     let pipeline_cache = self
        //         .create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder(), None)
        //         .unwrap();
        //
        //     let pipeline_shader_info = vk::PipelineShaderStageCreateInfo::builder()
        //         .stage(vk::ShaderStageFlags::COMPUTE)
        //         .module(shader)
        //         .name(CStr::from_bytes_with_nul(b"main\0").unwrap());
        //
        //     let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
        //         .stage(pipeline_shader_info.build())
        //         .layout(pipeline_layout);
        //     let compute_pipeline = self
        //         .create_compute_pipelines(pipeline_cache, &[compute_pipeline_info.build()], None)
        //         .unwrap()[0];
        //
        //     // Create Buffer Updates
        //     let desc_buffer_infos = buffers
        //         .iter()
        //         .map(|buffer| vk::DescriptorBufferInfo {
        //             buffer: buffer.buffer.buffer(),
        //             offset: 0,
        //             range: buffer.buffer.info().size as _,
        //         })
        //         .collect::<Vec<_>>();
        //     let write_desc_sets = [vk::WriteDescriptorSet::builder()
        //         .dst_set(desc_sets[0])
        //         .buffer_info(&desc_buffer_infos)
        //         .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        //         .dst_binding(0)
        //         .build()];
        //
        //     self.update_descriptor_sets(&write_desc_sets, &[]);
        //
        //     self.submit_global(|device, cb| {
        //         device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, compute_pipeline);
        //         device.cmd_bind_descriptor_sets(
        //             cb,
        //             vk::PipelineBindPoint::COMPUTE,
        //             pipeline_layout,
        //             0,
        //             &desc_sets,
        //             &[],
        //         );
        //         device.cmd_dispatch(cb, num as _, 1, 1);
        //     });
        // }

        // WARN: Very leaky function
        // TODO: Fix

        Ok(())
    }
}

pub struct VulkanBuffer {
    buffer: Buffer,
    device: VulkanDevice,
}

impl Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("size", &self.buffer.info().size)
            .finish()
    }
}

impl backend::BackendBuffer for VulkanBuffer {
    type Device = VulkanDevice;

    fn to_host<T: bytemuck::Pod>(&self) -> backend::Result<Vec<T>> {
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
        Ok(bytemuck::cast_slice(staging.mapped_slice()).to_vec())
    }

    fn size(&self) -> usize {
        self.buffer.info().size
    }
    fn device(&self) -> &Self::Device {
        &self.device
    }
}
