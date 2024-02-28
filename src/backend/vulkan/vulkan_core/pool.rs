use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use super::device::{self, Device};

type Cache<R> = Arc<Mutex<Vec<R>>>;

pub trait Resource {
    type Info: Hash + Eq + Clone;
    fn create(device: &Device, info: &Self::Info) -> Self;
    fn destroy(&mut self, device: &device::InternalDevice) {}
}

pub struct ResourcePool<R: Resource> {
    pub resources: Mutex<HashMap<R::Info, Cache<R>>>,
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
        self.cache.lock().unwrap().push(resource);
    }
}

impl<R: Resource> Default for ResourcePool<R> {
    fn default() -> Self {
        Self {
            resources: Default::default(),
        }
    }
}

impl<R: Resource> ResourcePool<R> {
    pub fn lease(&self, device: &Device, info: &R::Info) -> Lease<R> {
        let mut resources = self.resources.lock().unwrap();
        let cache = resources
            .entry(info.clone())
            .or_insert(Arc::new(Mutex::new(Vec::with_capacity(1))));
        let resource = cache
            .lock()
            .unwrap()
            .pop()
            .map(|r| r)
            .unwrap_or_else(|| R::create(device, &info));

        Lease {
            resource: Some(resource),
            cache: cache.clone(),
        }
    }
    pub fn clear(&self, device: &device::InternalDevice) {
        for cache in self.resources.lock().unwrap().values() {
            for res in cache.lock().unwrap().iter_mut() {
                res.destroy(device);
            }
            cache.lock().unwrap().clear();
        }
    }
}
