use std::any::Any;
use std::hash::{Hash, Hasher};

use hephaestus as hep;
use jit::DynHash;

pub trait BSDF: hep::Traverse + DynHash {
    fn eval(&self);
}

impl Hash for dyn BSDF {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id().hash(state);
        self.dyn_hash(state)
    }
}
