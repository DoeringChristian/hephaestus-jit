use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;

use super::buffer;
use super::buffer::Buffer;
use super::buffer::BufferInfo;
use super::Device;
use super::VulkanDevice;
use ash::vk;

type Cache<R> = Rc<RefCell<Vec<R>>>;

pub trait Resource {
    type Info: Hash + Eq + Clone;
    fn create(device: &Device, info: &Self::Info) -> Self;
}

pub struct Lease<R: Resource> {
    resource: Option<R>,
    cache: Cache<R>,
}
impl<R: Resource> Deref for Lease<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.resource.as_ref().unwrap()
    }
}
impl<R: Resource> DerefMut for Lease<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resource.as_mut().unwrap()
    }
}

pub struct ResourcePool<R: Resource> {
    pub resources: HashMap<R::Info, Cache<R>>,
}
impl<R: Resource> Default for ResourcePool<R> {
    fn default() -> Self {
        Self {
            resources: Default::default(),
        }
    }
}

impl<R: Resource> ResourcePool<R> {
    fn lease(&mut self, device: &Device, info: &R::Info) -> Lease<R> {
        let cache = self
            .resources
            .entry(info.clone())
            .or_insert(Rc::new(RefCell::new(Vec::with_capacity(1))));
        let resource = cache
            .borrow_mut()
            .pop()
            .map(|r| r)
            .unwrap_or_else(|| R::create(device, &info));

        Lease {
            resource: Some(resource),
            cache: cache.clone(),
        }
    }
}

impl Resource for Buffer {
    type Info = BufferInfo;

    fn create(device: &Device, info: &Self::Info) -> Self {
        Self::create(device, info.clone())
    }
}

pub struct Pool {
    pub device: Device,
    pub buffers: ResourcePool<Buffer>,
    pub image_views: Vec<vk::ImageView>,
}

impl Deref for Pool {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Pool {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            buffers: Default::default(),
            image_views: vec![],
        }
    }
    pub fn buffer(&mut self, info: buffer::BufferInfo) -> Lease<Buffer> {
        self.buffers.lease(&self.device, &info)
    }
    pub fn image_view(&mut self, info: &vk::ImageViewCreateInfo) -> vk::ImageView {
        unsafe { self.device.create_image_view(info, None).unwrap() }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe {
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view, None);
            }
        }
    }
}
