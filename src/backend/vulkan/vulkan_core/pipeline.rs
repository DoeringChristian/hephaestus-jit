use std::any::{Any, TypeId};
// TODO: Unify precompiled and IR Pipeline workflow
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::CStr;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use super::acceleration_structure::AccelerationStructure;
use super::buffer::Buffer;
use super::device::Device;
use super::graph::RGraphPool;
use super::image::{Image, ImageViewInfo};
use ash::vk;

pub struct Pipeline {
    device: Arc<Device>,
    pipeline: Arc<InternalPipeline>,
}
impl Deref for Pipeline {
    type Target = InternalPipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl Pipeline {
    pub fn create<D: PipelineDef>(device: &Arc<Device>, def: D) -> Arc<Self> {
        let mut hasher = DefaultHasher::new();
        // Any::type_id(def).hash(&mut hasher);
        // TypeId::of::<D>().hash(&mut hasher);
        def.hash(&mut hasher);
        let hash = hasher.finish();

        let mut cache = device.pipeline_cache.lock().unwrap();
        let pipeline = cache.entry(hash).or_insert_with(move || {
            let info = def.generate();
            Arc::new(InternalPipeline::create(device, &info))
        });
        Arc::new(Self {
            device: device.clone(),
            pipeline: pipeline.clone(),
        })
    }
    pub fn vk(&self) -> vk::Pipeline {
        self.pipeline.pipeline
    }

    pub fn submit(
        &self,
        cb: vk::CommandBuffer,
        pool: &mut RGraphPool,
        device: &Device,
        write_sets: &[WriteSet],
        extent: (u32, u32, u32),
    ) {
        let desc_sets = pool.desc_sets(&self.desc_set_layouts);
        log::trace!("Recording Pipeline pass with extent {extent:?}");

        let buffer_infos = write_sets
            .iter()
            .map(|write_set| {
                write_set
                    .buffers
                    .iter()
                    .map(|info| vk::DescriptorBufferInfo {
                        buffer: info.buffer.vk(),
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                        // range: info.buffer.info().size as _,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let write_desc_sets = write_sets
            .iter()
            .enumerate()
            .map(|(i, write_set)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_sets[write_set.set as usize])
                    .buffer_info(&buffer_infos[i])
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_binding(write_set.binding)
            })
            .collect::<Vec<_>>();
        unsafe {
            log::trace!("Updating Descriptor Sets{write_desc_sets:#?}");
            device.update_descriptor_sets(&write_desc_sets, &[]);

            log::trace!("Binding Pipeline.");
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.vk());
            log::trace!("Binding Descriptor Sets.");
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &desc_sets,
                &[],
            );
            log::trace!("Dispatching Pipeline.");
            device.cmd_dispatch(cb, extent.0, extent.1, extent.2);
        }
    }
    pub fn submit_to_cbuffer(
        &self,
        cb: vk::CommandBuffer,
        pool: &mut RGraphPool,
        num: usize,
        buffers: &[Arc<Buffer>],
        images: &[Arc<Image>],
        accels: &[Arc<AccelerationStructure>],
    ) {
        let desc_sets = pool.desc_sets(&self.desc_set_layouts).to_vec();
        let image_views = images
            .iter()
            .map(|image| {
                image.image_view(ImageViewInfo {
                    ty: match image.info().ty {
                        vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
                        vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
                        vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
                        _ => todo!(),
                    },
                    format: image.info().format,

                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
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
                buffer: buffer.vk(),
                offset: 0,
                // range: buffer.info().size as _,
                range: vk::WHOLE_SIZE,
            })
            .collect::<Vec<_>>();

        let acceleration_structures = accels.iter().map(|accel| accel.accel).collect::<Vec<_>>();

        let mut desc_accel_infos = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&acceleration_structures);

        let write_desc_sets = [
            if !desc_buffer_infos.is_empty() {
                Some(
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_sets[0])
                        .buffer_info(&desc_buffer_infos)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(0),
                )
            } else {
                None
            },
            if !desc_image_infos.is_empty() {
                Some(
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_sets[0])
                        .image_info(&desc_image_infos)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_binding(1),
                )
            } else {
                None
            },
            if !acceleration_structures.is_empty() {
                let mut write_desc_set = vk::WriteDescriptorSet::default()
                    .dst_set(desc_sets[0])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .push_next(&mut desc_accel_infos);
                write_desc_set.descriptor_count = acceleration_structures.len() as _; // WARN: no
                                                                                      // Idea if this is correct
                Some(write_desc_set)
            } else {
                None
            },
        ]
        .into_iter()
        .flat_map(|s| s)
        .collect::<Vec<_>>();

        unsafe {
            self.device.update_descriptor_sets(&write_desc_sets, &[]);

            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.vk());
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &desc_sets,
                &[],
            );
            self.device.cmd_dispatch(cb, num as _, 1, 1);
        }
    }
}

#[derive(Debug)]
pub struct InternalPipeline {
    desc_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

// impl Drop for Pipeline {
//     fn drop(&mut self) {
//         unsafe {
//             self.device
//                 .destroy_pipeline_layout(self.pipeline_layout, None);
//             self.device.destroy_pipeline(self.pipeline, None);
//             // self.device.destroy_descriptor_pool(self.desc_pool, None);
//             for desc_set_layout in self.desc_set_layouts.iter() {
//                 self.device
//                     .destroy_descriptor_set_layout(*desc_set_layout, None);
//             }
//         }
//     }
// }

impl InternalPipeline {
    pub fn create(device: &Device, info: &PipelineInfo) -> Self {
        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::default().code(&info.code);
            let shader = device.create_shader_module(&shader_info, None).unwrap();

            // Create Descriptor Pool
            // let desc_sizes = [vk::DescriptorPoolSize {
            //     ty: vk::DescriptorType::STORAGE_BUFFER,
            //     descriptor_count: 2 ^ 16,
            // }];

            // Create Layout
            let desc_set_layouts = info
                .desc_set_layouts
                .iter()
                .map(|desc_set_layout| {
                    let desc_layout_bindings = desc_set_layout
                        .bindings
                        .iter()
                        .map(|binding| vk::DescriptorSetLayoutBinding {
                            binding: binding.binding,
                            descriptor_type: binding.ty,
                            descriptor_count: binding.count as _,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>();
                    let desc_info = vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&desc_layout_bindings);
                    device
                        .create_descriptor_set_layout(&desc_info, None)
                        .unwrap()
                })
                .collect::<Vec<_>>();

            // Create Pipeline
            let pipeline_layout_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&desc_set_layouts);
            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap();
            let pipeline_cache = device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                .unwrap();

            let pipeline_shader_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(CStr::from_bytes_with_nul(b"main\0").unwrap());

            let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(pipeline_shader_info)
                .layout(pipeline_layout);
            let compute_pipeline = device
                .create_compute_pipelines(pipeline_cache, &[compute_pipeline_info], None)
                .unwrap()[0];

            // Destruct temporary elements
            device.destroy_shader_module(shader, None);
            device.destroy_pipeline_cache(pipeline_cache, None);

            Self {
                pipeline_layout,
                pipeline: compute_pipeline,
                desc_set_layouts,
            }
        }
    }
    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            // self.device.destroy_descriptor_pool(self.desc_pool, None);
            for desc_set_layout in self.desc_set_layouts.iter() {
                device.destroy_descriptor_set_layout(*desc_set_layout, None);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct Binding {
    pub binding: u32,
    pub count: u32,
    pub ty: vk::DescriptorType,
}

#[derive(Debug, Hash, Default)]
pub struct DescSetLayout {
    pub bindings: Vec<Binding>,
}

#[derive(Debug, Hash)]
pub struct PipelineInfo {
    pub code: Box<[u32]>,
    pub desc_set_layouts: Box<[DescSetLayout]>,
}
impl PipelineInfo {
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        <Self as Hash>::hash(self, &mut hasher);
        hasher.finish()
    }
}
pub trait PipelineDef: Hash {
    fn generate(self) -> PipelineInfo;
}
impl PipelineDef for &[u32] {
    fn generate(self) -> PipelineInfo {
        let code: Box<[u32]> = self.into();
        let entry_points = spirq::ReflectConfig::new()
            .ref_all_rscs(true)
            .spv(&*code)
            .reflect()
            .unwrap();
        assert_eq!(entry_points.len(), 1);

        let entry_point = entry_points.into_iter().next().unwrap();

        // Hash set for deduplicating bindings
        let mut vars = HashMap::new();
        let max_descriptor_sets = entry_point
            .vars
            .iter()
            .map(|var| match var {
                spirq::var::Variable::Descriptor {
                    name,
                    desc_bind,
                    desc_ty,
                    ty,
                    nbind,
                } => {
                    vars.insert((desc_bind.set(), desc_bind.bind()), var);
                    desc_bind.set()
                }
                _ => todo!(),
            })
            .max()
            .unwrap();

        let mut descriptor_sets = (0..max_descriptor_sets + 1)
            .map(|_| DescSetLayout::default())
            .collect::<Box<[_]>>();

        fn binding(binding: u32, count: u32, desc_ty: &spirq::ty::DescriptorType) -> Binding {
            Binding {
                binding,
                count,
                ty: match desc_ty {
                    spirq::ty::DescriptorType::CombinedImageSampler() => {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    }
                    spirq::ty::DescriptorType::StorageBuffer(_) => {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                    spirq::ty::DescriptorType::AccelStruct() => {
                        vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
                    }
                    _ => todo!(),
                },
            }
        }

        for var in vars.into_values() {
            match var {
                spirq::var::Variable::Descriptor {
                    name,
                    desc_bind,
                    desc_ty,
                    ty,
                    nbind,
                } => {
                    descriptor_sets[desc_bind.set() as usize]
                        .bindings
                        .push(binding(desc_bind.bind(), *nbind, &desc_ty));
                }
                _ => todo!(),
            }
        }

        PipelineInfo {
            code,
            desc_set_layouts: descriptor_sets,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct BufferWriteInfo<'a> {
    pub buffer: &'a Buffer,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct WriteSet<'a> {
    pub set: u32,
    pub binding: u32,
    pub buffers: &'a [BufferWriteInfo<'a>],
}
