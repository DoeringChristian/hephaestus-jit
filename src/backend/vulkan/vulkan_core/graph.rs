use ash::vk;
use indexmap::IndexMap;

use super::acceleration_structure::AccelerationStructure;
use super::buffer::Buffer;
use super::device::Device;
use super::image::Image;
use std::sync::Arc;

pub trait Resource {
    fn transition(&self, cb: vk::CommandBuffer, old: Access, new: Access);
}

pub struct RGraph {
    passes: Vec<Pass>,
    // We deduplicate resources by the pointers to their Arcs
    // Could also do enum instead of dyn, allowing for Rcs as well
    resources: IndexMap<usize, Arc<dyn Resource>>,
}
impl RGraph {
    pub fn new(device: &Device) -> Self {
        Self {
            passes: vec![],
            resources: Default::default(),
        }
    }
    pub fn resource(&mut self, resource: impl Into<Arc<dyn Resource>>) -> ResourceId {
        let resource = resource.into();
        let key = Arc::as_ptr(&resource) as *const () as usize;
        let entry = self.resources.entry(key);
        let id = ResourceId(entry.index());
        entry.or_insert_with(|| resource.clone());
        id
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceId(usize);
#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Access {
    pub flags: vk::AccessFlags,
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags,
}
impl Default for Access {
    fn default() -> Self {
        Self {
            flags: vk::AccessFlags::NONE,
            layout: vk::ImageLayout::default(),
            stage: vk::PipelineStageFlags::ALL_COMMANDS,
        }
    }
}

impl From<vk::AccessFlags> for Access {
    fn from(value: vk::AccessFlags) -> Self {
        Access {
            flags: value,
            ..Default::default()
        }
    }
}

///
/// Represents a Pass
///
pub struct Pass {
    read: Vec<(ResourceId, Access)>,
    write: Vec<(ResourceId, Access)>,
    render_fn: Option<Box<dyn FnOnce(&Device, vk::CommandBuffer)>>,
}

#[derive(Clone, Copy)]
pub struct PassApi<'a> {
    pub cb: &'a vk::CommandBuffer,
    pub device: &'a Device,
}

pub struct PassBuilder<'a> {
    graph: &'a mut RGraph,
    read: Vec<(ResourceId, Access)>,
    write: Vec<(ResourceId, Access)>,
}

// Impl resource for all 3 resource types
impl Resource for Buffer {
    fn transition(&self, cb: vk::CommandBuffer, old: Access, new: Access) {
        let buffer_memory_barriers = &[vk::BufferMemoryBarrier {
            src_access_mask: old.flags,
            dst_access_mask: new.flags,
            buffer: self.vk(),
            offset: 0,
            size: self.size() as _,
            ..Default::default()
        }];

        unsafe {
            self.device().cmd_pipeline_barrier(
                cb,
                old.stage,
                new.stage,
                vk::DependencyFlags::empty(),
                &[],
                buffer_memory_barriers,
                &[],
            );
        }
    }
}
impl Resource for Image {
    fn transition(&self, cb: vk::CommandBuffer, old: Access, new: Access) {
        let image_memory_barriers = &[vk::ImageMemoryBarrier {
            src_access_mask: old.flags,
            dst_access_mask: new.flags,
            old_layout: old.layout,
            new_layout: new.layout,
            image: self.vk(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        }];

        unsafe {
            self.device().cmd_pipeline_barrier(
                cb,
                old.stage,
                new.stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                image_memory_barriers,
            );
        }
    }
}
impl Resource for AccelerationStructure {
    fn transition(&self, cb: vk::CommandBuffer, old: Access, new: Access) {
        let memory_barriers = &[vk::MemoryBarrier {
            src_access_mask: old.flags,
            dst_access_mask: new.flags,
            ..Default::default()
        }];
        unsafe {
            self.device().cmd_pipeline_barrier(
                cb,
                old.stage,
                new.stage,
                vk::DependencyFlags::empty(),
                memory_barriers,
                &[],
                &[],
            );
        }
    }
}

impl From<Buffer> for Arc<dyn Resource> {
    fn from(value: Buffer) -> Self {
        Arc::new(value)
    }
}

impl<'a> PassBuilder<'a> {
    pub fn read(mut self, resource: Arc<dyn Resource>, access: impl Into<Access>) -> Self {
        let id = self.graph.resource(resource);
        let access = access.into();
        self.read.push((id, access));
        self
    }
    pub fn write(mut self, resource: Arc<dyn Resource>, access: impl Into<Access>) -> Self {
        let id = self.graph.resource(resource);
        let access = access.into();
        self.write.push((id, access));
        self
    }
    pub fn record(self, f: impl FnOnce(&Device, vk::CommandBuffer) + 'static) -> PassId {
        let id = PassId(self.graph.passes.len());
        self.graph.passes.push(Pass {
            read: self.read,
            write: self.write,
            render_fn: Some(Box::new(f)),
        });
        id
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
            .map(|_| Access::default())
            .collect::<Vec<_>>();

        let resources = self
            .resources
            .into_iter()
            .map(|(_, r)| r)
            .collect::<Vec<_>>();

        device.submit_global(|device, cb| {
            for pass in self.passes {
                // Transition resources
                for (id, access) in pass.read {
                    let prev = resource_accesses[id.0];
                    if prev != access {
                        resources[id.0].transition(cb, prev, access);
                        log::trace!("Barrier from {prev:?} -> {read:?}", read = access);
                    }
                }

                // Record content of pass
                let render_fn = pass.render_fn.unwrap();
                render_fn(device, cb);

                // Modify resource_accesses when writing
                for (id, access) in pass.write {
                    resource_accesses[id.0] = access;
                }
            }
        });
    }
}
