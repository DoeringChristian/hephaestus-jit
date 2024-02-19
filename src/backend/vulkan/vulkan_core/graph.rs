use super::utils::RangeGroupBy;
use ash::vk;
use indexmap::IndexMap;
use itertools::Itertools;
use slice_group_by::GroupBy;

use crate::backend::PassReport;

use super::acceleration_structure::AccelerationStructure;
use super::buffer::Buffer;
use super::device::Device;
use super::image::Image;
use super::profiler::Profiler;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use vk_sync::cmd::pipeline_barrier;
use vk_sync::{AccessType, ImageLayout};

#[derive(Debug)]
struct ImageBarrier {
    pub prev: AccessType,
    pub next: AccessType,
    pub previous_layout: ImageLayout,
    pub next_layout: ImageLayout,
    pub discard_contents: bool,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: vk::Image,
    pub range: vk::ImageSubresourceRange,
}
#[derive(Debug)]
struct BufferBarrier {
    pub prev: AccessType,
    pub next: AccessType,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: vk::Buffer,
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug, Default)]
struct Barriers {
    prev: Vec<AccessType>,
    next: Vec<AccessType>,
    buffer_barriers: Vec<BufferBarrier>,
    image_barriers: Vec<ImageBarrier>,
}
impl Barriers {
    pub fn record(&self, device: &Device, cb: vk::CommandBuffer) {
        let buffer_barriers = self
            .buffer_barriers
            .iter()
            .map(|barrier| vk_sync::BufferBarrier {
                previous_accesses: std::slice::from_ref(&barrier.prev),
                next_accesses: std::slice::from_ref(&barrier.next),
                src_queue_family_index: barrier.src_queue_family_index,
                dst_queue_family_index: barrier.dst_queue_family_index,
                buffer: barrier.buffer,
                offset: barrier.offset,
                size: barrier.size,
            })
            .collect::<Vec<_>>();
        let image_barriers = self
            .image_barriers
            .iter()
            .map(|barrier| vk_sync::ImageBarrier {
                previous_accesses: std::slice::from_ref(&barrier.prev),
                next_accesses: std::slice::from_ref(&barrier.next),
                previous_layout: barrier.previous_layout,
                next_layout: barrier.next_layout,
                discard_contents: barrier.discard_contents,
                src_queue_family_index: barrier.src_queue_family_index,
                dst_queue_family_index: barrier.dst_queue_family_index,
                image: barrier.image,
                range: barrier.range,
            })
            .collect::<Vec<_>>();

        let global_barrier =
            (self.prev.is_empty() && self.next.is_empty()).then(|| vk_sync::GlobalBarrier {
                previous_accesses: &self.prev,
                next_accesses: &self.next,
            });
        log::trace!("Barrier with Global {global_barrier:?}, Buffer {buffer_barriers:?}, and Image {image_barriers:?}");

        pipeline_barrier(
            device,
            cb,
            global_barrier,
            &buffer_barriers,
            &image_barriers,
        );
    }
    pub fn global_barrier(&mut self, prev: AccessType, next: AccessType) {
        self.prev.push(prev);
        self.next.push(next);
    }
    pub fn buffer_barrier(&mut self, barrier: BufferBarrier) {
        self.buffer_barriers.push(barrier)
    }
    pub fn image_barrierr(&mut self, barrier: ImageBarrier) {
        self.image_barriers.push(barrier)
    }
}

#[derive(Debug)]
pub enum Resource {
    Buffer(Arc<Buffer>),
    Image(Arc<Image>),
    AccelerationStructure(Arc<AccelerationStructure>),
}
impl Resource {
    fn transition(&self, barriers: &mut Barriers, prev: AccessType, next: AccessType) {
        // TODO: Image Layout transitions
        match self {
            Resource::Buffer(buffer) => barriers.buffer_barrier(BufferBarrier {
                prev,
                next,
                src_queue_family_index: 0,
                dst_queue_family_index: 0,
                buffer: buffer.vk(),
                offset: 0,
                size: buffer.size(),
            }),
            Resource::Image(image) => barriers.image_barrierr(ImageBarrier {
                prev,
                next,
                previous_layout: ImageLayout::Optimal,
                next_layout: ImageLayout::Optimal,
                discard_contents: prev == AccessType::Nothing,
                src_queue_family_index: 0,
                dst_queue_family_index: 0,
                image: image.vk(),
                range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
            }),
            Resource::AccelerationStructure(acceleration_structure) => {
                barriers.global_barrier(prev, next);
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

/// The simple most Render Graph implementation I could come up with.
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
impl Debug for RGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RGraph")
            .field("passes", &self.passes)
            .field("resources", &self.resources)
            .finish()
    }
}
impl RGraph {
    pub fn new() -> Self {
        Self {
            // profiler: ProfilerData::new(&device.device, ProfilerBackend::new(&device)),
            passes: Default::default(),
            resources: Default::default(),
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(usize);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PassId(usize);

///
/// Represents a recorded Pass
///
pub struct Pass {
    name: String,
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
    name: &'a str,
    graph: &'a mut RGraph,
    read: Vec<(ResourceId, AccessType)>,
    write: Vec<(ResourceId, AccessType)>,
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
            name: self.name.into(),
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
        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&desc_sizes)
            .max_sets(set_layouts.len() as _);
        let desc_pool = unsafe {
            self.device
                .create_descriptor_pool(&desc_pool_info, None)
                .unwrap()
        };
        self.desc_pools.push(desc_pool);

        let desc_set_allocation_info = vk::DescriptorSetAllocateInfo::default()
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
    pub fn pass<'a>(&'a mut self, name: &'a str) -> PassBuilder<'a> {
        PassBuilder {
            name,
            graph: self,
            read: vec![],
            write: vec![],
        }
    }
    pub fn submit(mut self, device: &Device) -> (std::time::Duration, Vec<PassReport>) {
        // Passes are already in topological order
        //
        log::trace!("Passes: {passes:#?}", passes = self.passes);

        let mut deps: Vec<PassId> = vec![];
        let mut pass_deps = vec![];
        let mut last_writes = vec![None; self.resources.len()];
        let mut dep_counts = vec![0u32; self.passes.len()];

        for id in (0..self.passes.len()).map(|i| PassId(i)) {
            let pass = &self.passes[id.0];
            let start = deps.len();

            // Get the passes this pass is dependent on.
            deps.extend(
                pass.read
                    .iter()
                    .map(|(id, _)| *id)
                    .chain(pass.write.iter().map(|(id, _)| *id))
                    .unique()
                    .flat_map(|r| last_writes[r.0]),
            );
            let range = start..deps.len();
            pass_deps.push(range.clone());

            // Increment dependant count for this pass
            for dep in &deps[range] {
                dep_counts[dep.0] += 1;
            }

            // Update the last_writes field
            for (r, _) in &pass.write {
                last_writes[r.0] = Some(id);
            }
        }

        dbg!(&deps);
        dbg!(&pass_deps);
        dbg!(&last_writes);
        dbg!(&dep_counts);

        let mut frontier = dep_counts
            .iter()
            .enumerate()
            .filter(|(i, count)| **count == 0)
            .map(|(i, _)| PassId(i))
            .collect::<Vec<_>>();
        dbg!(&frontier);
        let mut new_frontier: Vec<PassId> = vec![];

        let mut passes = vec![];
        let mut groups = vec![];

        while passes.len() < self.passes.len() {
            // Iterate through all dependencies of all variables in the frontier
            for &id in frontier
                .iter()
                .flat_map(|id| deps[pass_deps[id.0].clone()].iter())
            {
                dep_counts[id.0] -= 1;
            }
            // Collect the new frontier by searching for all 0 dependant variables
            // These have to be among the dependencies of the current frontier
            new_frontier.extend(
                frontier
                    .iter()
                    .flat_map(|id| deps[pass_deps[id.0].clone()].iter())
                    .filter(|id| dep_counts[id.0] == 0)
                    .unique(),
            );

            let start = passes.len();
            passes.extend(frontier.drain(..));
            groups.push(start..passes.len());

            // Swap out the now empty frontier with the new frontier
            std::mem::swap(&mut frontier, &mut new_frontier);
            assert!(new_frontier.is_empty());
        }

        // let mut prev_passes = self
        //     .passes
        //     .into_iter()
        //     .map(|pass| Some(pass))
        //     .collect::<Vec<_>>();
        // let passes = passes
        //     .iter()
        //     .map(|id| prev_passes[id.0].take().unwrap())
        //     .collect::<Vec<_>>();

        // NOTE: groups and passes are now in reverse order
        // We could flip both, but since the order of passes in a group are per definition
        // interchangable, we are able to only flip groups

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

        let mut profiler = Profiler::new(device, passes.len());
        let mut pass_names = vec![];

        let cpu_time = device.submit_global(|device, cb| {
            profiler.begin_frame(cb);

            for group in groups {
                let scope = profiler.begin_scope(cb);

                let group_name = Itertools::intersperse(
                    passes[group.clone()]
                        .into_iter()
                        .map(|pass| pass.name.as_str()),
                    ", ",
                )
                .collect::<String>();

                log::trace!(
                    "Recording passes {passes:?} in group \"{group_name}\"",
                    passes = group.clone()
                );

                // Transition resources
                // TODO: Improved barrier placement
                // Also verify correcness especially for write after write
                // Could also make the rg use ssa to optimize?
                // log::trace!("Recording {pass:?} to command buffer");

                let mut barriers = Barriers::default();

                // Record barriers for all the passes in a group by transitioning their states
                for (id, access) in passes[group.clone()]
                    .iter()
                    .flat_map(|pass| pass.read.iter())
                    .chain(
                        passes[group.clone()]
                            .iter()
                            .flat_map(|pass| pass.write.iter()),
                    )
                {
                    let prev = &mut resource_accesses[id.0];
                    if *prev != *access {
                        resources[id.0].transition(&mut barriers, *prev, *access);
                        log::trace!(
                            "\tTransition Resource {resource:?} {prev:?} -> {read:?}",
                            resource = id.0,
                            read = access
                        );
                        *prev = *access;
                    }
                }
                barriers.record(device, cb);

                // Record content of pass
                for pass in &mut passes[group] {
                    let render_fn = pass.render_fn.take().unwrap();
                    render_fn(device, cb, &mut tmp_resource_pool);
                }

                profiler.end_scope(cb, scope);

                pass_names.push(group_name);
            }

            profiler.end_frame(cb);
        });

        let report = profiler
            .report()
            .into_iter()
            .zip(pass_names)
            .map(|(duration, name)| PassReport { name, duration })
            .collect::<Vec<_>>();

        drop(tmp_resource_pool);

        (cpu_time, report)
    }
}
