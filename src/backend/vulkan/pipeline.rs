// TODO: Unify precompiled and IR Pipeline workflow
use std::collections::hash_map::DefaultHasher;
use std::ffi::CStr;
use std::hash::{Hash, Hasher};

use crate::backend::vulkan::codegen;
use crate::ir::IR;

use super::buffer::Buffer;
use super::device::Device;
use super::image::Image;
use ash::vk;

#[derive(Debug)]
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
    pub fn create<'a>(device: &Device, desc: &PipelineDesc<'a>) -> Self {
        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::builder()
                .code(desc.code)
                .build();
            let shader = device.create_shader_module(&shader_info, None).unwrap();

            // Create Descriptor Pool
            let desc_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2 ^ 16,
            }];
            let desc_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&desc_sizes)
                .max_sets(desc.desc_set_layouts.len() as _);
            let desc_pool = device
                .create_descriptor_pool(&desc_pool_info, None)
                .unwrap();

            // Create Layout
            let desc_set_layouts = desc
                .desc_set_layouts
                .iter()
                .map(|desc_set_layout| {
                    let desc_layout_bindings = desc_set_layout
                        .bindings
                        .iter()
                        .map(|binding| vk::DescriptorSetLayoutBinding {
                            binding: binding.binding,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: binding.count as _,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>();
                    let desc_info = vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&desc_layout_bindings);
                    device
                        .create_descriptor_set_layout(&desc_info, None)
                        .unwrap()
                })
                .collect::<Vec<_>>();

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
    pub fn from_ir(device: &Device, ir: &IR) -> Self {
        let spirv = codegen::assemble_trace(ir, "main").unwrap();

        let num_buffers = ir.n_buffers;
        let num_textures = ir.n_textures;

        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::builder()
                .code(spirv.as_slice())
                .build();
            let shader = device.create_shader_module(&shader_info, None).unwrap();

            // Create Descriptor Pool
            let desc_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: num_buffers as _,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: num_textures as _,
                },
            ];
            let desc_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&desc_sizes)
                .max_sets(1);
            let desc_pool = device
                .create_descriptor_pool(&desc_pool_info, None)
                .unwrap();

            // Create Layout
            let desc_layout_bindings = [
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: num_buffers as _,
                    stage_flags: vk::ShaderStageFlags::ALL,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: num_textures as _,
                    stage_flags: vk::ShaderStageFlags::ALL,
                    ..Default::default()
                },
            ];
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
    pub fn submit(
        &self,
        cb: vk::CommandBuffer,
        device: &Device,
        write_sets: &[WriteSet],
        extent: (u32, u32, u32),
    ) {
        let buffer_infos = write_sets
            .iter()
            .map(|write_set| {
                write_set
                    .buffers
                    .iter()
                    .map(|info| vk::DescriptorBufferInfo {
                        buffer: info.buffer.buffer(),
                        offset: 0,
                        range: info.buffer.info().size as _,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let write_desc_sets = write_sets
            .iter()
            .enumerate()
            .map(|(i, write_set)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.desc_sets[write_set.set as usize])
                    .buffer_info(&buffer_infos[i])
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_binding(0)
                    .build()
            })
            .collect::<Vec<_>>();
        unsafe {
            log::trace!("Updating Descriptor Sets{write_desc_sets:#?}");
            self.device.update_descriptor_sets(&write_desc_sets, &[]);

            log::trace!("Binding Pipeline.");
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            log::trace!("Binding Descriptor Sets.");
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &self.desc_sets,
                &[],
            );
            log::trace!("Dispatching Pipeline.");
            device.cmd_dispatch(cb, extent.0, extent.1, extent.2);
        }
    }
    pub fn submit_to_cbuffer<'a>(
        &'a self,
        cb: vk::CommandBuffer,
        device: &Device,
        num: usize,
        buffers: &[&Buffer],
        images: &[&Image],
    ) {
        let image_views = images
            .iter()
            .map(|image| {
                let info = vk::ImageViewCreateInfo::builder()
                    .image(***image)
                    .view_type(match image.info().ty {
                        vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
                        vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
                        vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
                        _ => todo!(),
                    })
                    .format(image.info().format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&info, None).unwrap() }
            })
            .collect::<Vec<_>>();

        let desc_image_infos = images
            .iter()
            .enumerate()
            .map(|(i, image)| vk::DescriptorImageInfo {
                sampler: image.default_sampler(),
                image_view: image_views[i],
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .collect::<Vec<_>>();

        let desc_buffer_infos = buffers
            .iter()
            .map(|buffer| vk::DescriptorBufferInfo {
                buffer: buffer.buffer(),
                offset: 0,
                range: buffer.info().size as _,
            })
            .collect::<Vec<_>>();
        let write_desc_sets = [
            vk::WriteDescriptorSet::builder()
                .dst_set(self.desc_sets[0])
                .buffer_info(&desc_buffer_infos)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .dst_binding(0)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.desc_sets[0])
                .image_info(&desc_image_infos)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(1)
                .build(),
        ];

        unsafe {
            self.device.update_descriptor_sets(&write_desc_sets, &[]);

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
        }
    }
    // pub fn launch_fenced<'a>(&'a self, num: usize, buffers: impl Iterator<Item = &'a Buffer>) {
    //     self.device.submit_global(|device, cb| {
    //         self.submit_to_cbuffer(cb, device, num, buffers, [].iter());
    //     })
    // }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct Binding {
    pub binding: u32,
    pub count: u32,
    // pub ty: vk::DescriptorType,
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct DescSetLayout<'a> {
    pub bindings: &'a [Binding],
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct PipelineDesc<'a> {
    pub code: &'a [u32],
    pub desc_set_layouts: &'a [DescSetLayout<'a>],
}
impl<'a> PipelineDesc<'a> {
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        <Self as Hash>::hash(self, &mut hasher);
        hasher.finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferWriteInfo<'a> {
    pub buffer: &'a Buffer,
}

#[derive(Debug, Clone, Copy)]
pub struct WriteSet<'a> {
    pub set: u32,
    pub binding: u32,
    pub buffers: &'a [BufferWriteInfo<'a>],
}
