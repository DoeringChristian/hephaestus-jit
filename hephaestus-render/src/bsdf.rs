use std::any::Any;
use std::hash::{Hash, Hasher};

use hep::*;
use hephaestus as hep;
use jit::DynHash;

pub trait BSDF: Traverse + DynHash {
    fn eval(&self, wi: Vector3f, wo: Vector3f, active: Mask) -> Vector3f;
}

impl Hash for dyn BSDF {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id().hash(state);
        self.dyn_hash(state)
    }
}

#[derive(Hash, Clone, Traverse, Construct)]
pub struct DiffuseBSDF {
    pub albedo: Vector3f,
}

impl BSDF for DiffuseBSDF {
    fn eval(&self, wi: Vector3f, wo: Vector3f, active: Mask) -> Vector3f {
        self.albedo.clone()
    }
}
