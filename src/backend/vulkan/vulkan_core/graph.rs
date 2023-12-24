use ash::vk;
use indexmap::IndexMap;

use super::acceleration_structure::AccelerationStructure;
use super::buffer::Buffer;
use super::device::Device;
use super::image::Image;
use std::fmt::Debug;
use std::sync::Arc;
use vk_sync::{cmd::pipeline_barrier, BufferBarrier, GlobalBarrier, ImageBarrier};
use vk_sync::{AccessType, ImageLayout};

pub trait ResourceTrait: Debug {
    fn transition(&self, cb: vk::CommandBuffer, prev: AccessType, next: AccessType);
}

#[derive(Debug)]
pub enum Resource {
    Buffer(Arc<Buffer>),
    Image(Arc<Image>),
    AccelerationStructure(Arc<AccelerationStructure>),
}
impl Resource {
    pub fn transition(&self, cb: vk::CommandBuffer, prev: AccessType, next: AccessType) {
        match self {
            Resource::Buffer(buffer) => buffer.transition(cb, prev, next),
            Resource::Image(image) => image.transition(cb, prev, next),
            Resource::AccelerationStructure(acceleration_structure) => {
                acceleration_structure.transition(cb, prev, next)
            }
        }
    }
}

impl From<Arc<Buffer>> for Resource {
    fn from(value: Arc<Buffer>) -> Self {
        Resource::Buffer(value)
    }
}
impl From<Arc<Image>> for Resource {
    fn from(value: Arc<Image>) -> Self {
        Resource::Image(value)
    }
}
impl From<Arc<AccelerationStructure>> for Resource {
    fn from(value: Arc<AccelerationStructure>) -> Self {
        Resource::AccelerationStructure(value)
    }
}

#[derive(Default, Debug)]
/// The most simpel Render Graph implementation I could come up with.
/// It is not efficient (it heavily relies on Rust's smart pointers)
///
/// * `passes`: A number of passes, that can be recorded into a command buffer
/// * `resources`: A hashmap mapping from the pointer addresses of resources to their owned types
/// usize -> (Arc<dyn Resouce>)
/// This let's us deduplicate resources (a better mechanism wouls be use indices)
pub struct RGraph {
    passes: Vec<Pass>,
    // We deduplicate resources by the pointers to their Arcs
    // Could also do enum instead of dyn, allowing for Rcs as well
    // resources: IndexMap<usize, Arc<dyn Resource>>,
    resources: IndexMap<usize, Resource>,
}
impl RGraph {
    pub fn resource<R>(&mut self, resource: &Arc<R>) -> ResourceId
    where
        Arc<R>: Into<Resource>,
    {
        let key = Arc::as_ptr(&resource) as *const () as usize;
        let entry = self.resources.entry(key);
        let id = ResourceId(entry.index());
        entry.or_insert_with(|| resource.clone().into());
        id
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceId(usize);
#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);

///
/// Represents a recorded Pass
///
pub struct Pass {
    read: Vec<(ResourceId, AccessType)>,
    write: Vec<(ResourceId, AccessType)>,
    render_fn: Option<Box<dyn FnOnce(&Device, vk::CommandBuffer, &mut RGraphPool)>>,
}
impl Debug for Pass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pass")
            .field("read", &self.read)
            .field("write", &self.write)
            .finish()
    }
}

/// Builder, used to construct a pass.
///
/// * `graph`: The graph on which the pass is constructed
/// * `read`: Read accesses
/// * `write`: Write accesses
pub struct PassBuilder<'a> {
    graph: &'a mut RGraph,
    read: Vec<(ResourceId, AccessType)>,
    write: Vec<(ResourceId, AccessType)>,
}

// Impl resource for all 3 resource types
impl ResourceTrait for Buffer {
    fn transition(&self, cb: vk::CommandBuffer, prev: AccessType, next: AccessType) {
        pipeline_barrier(
            self.device(),
            cb,
            None,
            &[BufferBarrier {
                previous_accesses: &[prev],
                next_accesses: &[next],
                src_queue_family_index: 0,
                dst_queue_family_index: 0,
                buffer: self.vk(),
                offset: 0,
                size: self.size(),
            }],
            &[],
        );
    }
}
impl ResourceTrait for Image {
    fn transition(&self, cb: vk::CommandBuffer, prev: AccessType, next: AccessType) {
        pipeline_barrier(
            self.device(),
            cb,
            None,
            &[],
            &[ImageBarrier {
                previous_accesses: &[prev],
                next_accesses: &[next],
                previous_layout: ImageLayout::Optimal,
                next_layout: ImageLayout::Optimal,
                discard_contents: prev == AccessType::Nothing,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.vk(),
                range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
            }],
        )
    }
}
impl ResourceTrait for AccelerationStructure {
    fn transition(&self, cb: vk::CommandBuffer, prev: AccessType, next: AccessType) {
        pipeline_barrier(
            self.device(),
            cb,
            Some(GlobalBarrier {
                previous_accesses: &[prev],
                next_accesses: &[next],
            }),
            &[],
            &[],
        )
    }
}

impl From<Buffer> for Arc<dyn ResourceTrait> {
    fn from(value: Buffer) -> Self {
        Arc::new(value)
    }
}

impl<'a> PassBuilder<'a> {
    pub fn read<R>(mut self, resource: &Arc<R>, access: AccessType) -> Self
    where
        Arc<R>: Into<Resource>,
    {
        let id = self.graph.resource(resource);
        let access = access.into();
        self.read.push((id, access));
        self
    }
    pub fn write<R>(mut self, resource: &Arc<R>, access: AccessType) -> Self
    where
        Arc<R>: Into<Resource>,
    {
        let id = self.graph.resource(resource);
        let access = access.into();
        self.write.push((id, access));
        self
    }
    pub fn record(
        self,
        f: impl FnOnce(&Device, vk::CommandBuffer, &mut RGraphPool) + 'static,
    ) -> PassId {
        let id = PassId(self.graph.passes.len());
        self.graph.passes.push(Pass {
            read: self.read,
            write: self.write,
            render_fn: Some(Box::new(f)),
        });
        id
    }
}

/// A Resource Pool for temporary resources such as image views or descriptor sets
#[derive(Debug)]
pub struct RGraphPool {
    pub device: Device,
    pub desc_sets: Vec<vk::DescriptorSet>,
    pub desc_pools: Vec<vk::DescriptorPool>,
}

impl RGraphPool {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            // image_views: vec![],
            desc_sets: vec![],
            desc_pools: vec![],
        }
    }
    pub fn desc_sets(&mut self, set_layouts: &[vk::DescriptorSetLayout]) -> &[vk::DescriptorSet] {
        let desc_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2 ^ 16,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 2 ^ 16,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 2 ^ 16,
            },
        ];
        let desc_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&desc_sizes)
            .max_sets(set_layouts.len() as _);
        let desc_pool = unsafe {
            self.device
                .create_descriptor_pool(&desc_pool_info, None)
                .unwrap()
        };
        self.desc_pools.push(desc_pool);

        let desc_set_allocation_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(set_layouts);
        let desc_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&desc_set_allocation_info)
                .unwrap()
        };
        let start = self.desc_sets.len();
        self.desc_sets.extend_from_slice(&desc_sets);
        &self.desc_sets[start..]
    }
}

impl Drop for RGraphPool {
    fn drop(&mut self) {
        unsafe {
            for pool in self.desc_pools.drain(..) {
                self.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}

impl RGraph {
    pub fn pass(&mut self) -> PassBuilder {
        PassBuilder {
            graph: self,
            read: vec![],
            write: vec![],
        }
    }
    pub fn submit(self, device: &Device) {
        let mut resource_accesses = self
            .resources
            .iter()
            .map(|_| AccessType::Nothing)
            .collect::<Vec<_>>();

        let resources = self
            .resources
            .into_iter()
            .map(|(_, r)| r)
            .collect::<Vec<_>>();

        let mut tmp_resource_pool = RGraphPool::new(device);

        device.submit_global(|device, cb| {
            for pass in self.passes {
                // Transition resources
                log::trace!("Recording {pass:?} to command buffer");

                for (id, access) in pass.read.iter().chain(pass.write.iter()) {
                    let prev = resource_accesses[id.0];
                    if prev != *access {
                        resources[id.0].transition(cb, prev, *access);
                        log::trace!("Barrier from {prev:?} -> {read:?}", read = access);
                    }
                }

                // Record content of pass
                let render_fn = pass.render_fn.unwrap();
                render_fn(device, cb, &mut tmp_resource_pool);

                // Modify resource_accesses when writing
                for (id, access) in pass.write.iter() {
                    resource_accesses[id.0] = *access;
                }
            }
        });

        drop(tmp_resource_pool);
    }
}
