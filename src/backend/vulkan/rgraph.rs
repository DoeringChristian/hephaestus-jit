use std::ops::{Deref, DerefMut};

use super::buffer::{Buffer, BufferInfo};
use super::VulkanDevice;
use ash::vk;

pub struct Pass {
    read: Vec<ResourceId>,
    write: Vec<ResourceId>,
    render_fn: Option<Box<dyn FnOnce(&RGraph, vk::CommandBuffer)>>,
}

pub enum Resource {
    Buffer(Buffer),
    ImageView(vk::ImageView),
}

impl<'a> From<(&'a VulkanDevice, BufferInfo)> for Resource {
    fn from(value: (&'a VulkanDevice, BufferInfo)) -> Self {
        Self::Buffer(Buffer::create(&value.0.device, value.1))
    }
}
impl<'a> From<(&'a VulkanDevice, vk::ImageViewCreateInfo)> for Resource {
    fn from(value: (&'a VulkanDevice, vk::ImageViewCreateInfo)) -> Self {
        Self::ImageView(unsafe { value.0.device.create_image_view(&value.1, None).unwrap() })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceId(usize);

pub struct RGraph {
    device: VulkanDevice,
    resources: Vec<Resource>,
    passes: Vec<Pass>,
}
impl Deref for RGraph {
    type Target = VulkanDevice;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl RGraph {
    pub fn new(device: &VulkanDevice) -> Self {
        Self {
            device: device.clone(),
            resources: Default::default(),
            passes: Default::default(),
        }
    }
    pub fn pass(&mut self) -> PassBuilder {
        PassBuilder {
            graph: self,
            read: Default::default(),
            write: Default::default(),
        }
    }
    pub fn resource<'a, I>(&'a mut self, info: I) -> ResourceId
    where
        (&'a VulkanDevice, I): Into<Resource>,
    {
        let id = ResourceId(self.resources.len());
        let resource = (&self.device, info).into();
        self.resources.push(resource);
        id
    }
    pub fn buffer(&self, id: ResourceId) -> &Buffer {
        match &self.resources[id.0] {
            Resource::Buffer(buffer) => buffer,
            _ => todo!(),
        }
    }
    pub fn image_view(&self, id: ResourceId) -> &vk::ImageView {
        match &self.resources[id.0] {
            Resource::ImageView(view) => view,
            _ => todo!(),
        }
    }
    pub fn record_slow(mut self, cb: vk::CommandBuffer) {
        let passes = std::mem::take(&mut self.passes);
        for mut pass in passes {
            // let memory_barriers = [vk::MemoryBarrier::builder()
            //     .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            //     .dst_access_mask(vk::AccessFlags::SHADER_READ)
            //     .build()];
            // unsafe {
            //     self.device.cmd_pipeline_barrier(
            //         cb,
            //         vk::PipelineStageFlags::ALL_COMMANDS,
            //         vk::PipelineStageFlags::ALL_COMMANDS,
            //         vk::DependencyFlags::empty(),
            //         &memory_barriers,
            //         &[],
            //         &[],
            //     );
            // }
            pass.render_fn.take().unwrap()(&self, cb);
        }
        todo!()
    }
}

pub struct PassBuilder<'a> {
    graph: &'a mut RGraph,
    read: Vec<ResourceId>,
    write: Vec<ResourceId>,
}
impl<'a> PassBuilder<'a> {
    pub fn read(mut self, id: ResourceId) -> Self {
        self.read.push(id);
        self
    }
    pub fn write(mut self, id: ResourceId) -> Self {
        self.write.push(id);
        self
    }
    pub fn exec(self, f: impl FnOnce(&RGraph, vk::CommandBuffer) + 'static) {
        self.graph.passes.push(Pass {
            read: self.read,
            write: self.write,
            render_fn: Some(Box::new(f)),
        })
    }
}
impl<'a> Deref for PassBuilder<'a> {
    type Target = RGraph;

    fn deref(&self) -> &Self::Target {
        self.graph
    }
}
impl<'a> DerefMut for PassBuilder<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.graph
    }
}
