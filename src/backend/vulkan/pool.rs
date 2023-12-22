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

impl<R: Resource> Drop for Lease<R> {
    fn drop(&mut self) {
        let resource = self.resource.take().unwrap();
        self.cache.borrow_mut().push(resource);
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

// Simple Bump allocator esque pool
pub struct BumpPool<R: Resource> {
    pub cache: Cache<R>,
}

impl<R: Resource> Default for BumpPool<R> {
    fn default() -> Self {
        Self {
            cache: Rc::new(RefCell::new(vec![])),
        }
    }
}
impl<R: Resource> BumpPool<R> {
    fn lease(&mut self, device: &Device, info: &R::Info) -> Lease<R> {
        let resource = R::create(device, &info);

        Lease {
            resource: Some(resource),
            cache: self.cache.clone(),
        }
    }
}

pub struct Pool {
    pub device: Device,
    pub buffer_pool: ResourcePool<Buffer>,
    pub buffers: BumpPool<Buffer>,
    pub image_views: Vec<vk::ImageView>,
    pub desc_sets: Vec<vk::DescriptorSet>,
    pub desc_pools: Vec<vk::DescriptorPool>,
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
            buffer_pool: Default::default(),
            image_views: vec![],
            desc_sets: vec![],
            desc_pools: vec![],
            // desc_pool,
        }
    }
    pub fn lease_buffer(&mut self, info: buffer::BufferInfo) -> Lease<Buffer> {
        self.buffer_pool.lease(&self.device, &info)
    }
    pub fn buffer(&mut self, info: buffer::BufferInfo) -> Lease<Buffer> {
        self.buffers.lease(&self.device, &info)
    }
    pub fn image_view(&mut self, info: &vk::ImageViewCreateInfo) -> vk::ImageView {
        let view = unsafe { self.device.create_image_view(info, None).unwrap() };
        self.image_views.push(view);
        view
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

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe {
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view, None);
            }
            for pool in self.desc_pools.drain(..) {
                self.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}
