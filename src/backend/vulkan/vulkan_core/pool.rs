use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

type Cache<R> = Arc<Mutex<Vec<R>>>;

pub trait Resource {
    type Info: Hash + Eq + Clone;
    type Context;
    fn create(context: &Self::Context, info: Self::Info) -> Self;
}

pub struct ResourcePool<R: Resource> {
    pub resources: HashMap<R::Info, Cache<R>>,
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
    pub fn lease(&mut self, context: &R::Context, info: &R::Info) -> Lease<R> {
        let cache = self
            .resources
            .entry(info.clone())
            .or_insert(Arc::new(Mutex::new(Vec::with_capacity(1))));
        let resource = cache
            .lock()
            .unwrap()
            .pop()
            .map(|r| r)
            .unwrap_or_else(|| R::create(context, info.clone()));

        Lease {
            resource: Some(resource),
            cache: cache.clone(),
        }
    }
}
