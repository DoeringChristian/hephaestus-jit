use std::any::Any;
use std::hash::{Hash, Hasher};

use hephaestus as hep;
use jit::DynHash;

pub trait BSDF: hep::Traverse {
    fn eval(&self);
}

impl hep::Traverse for Box<dyn BSDF> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        self.as_ref().traverse(vars)
    }

    fn ravel(&self) -> jit::VarRef {
        self.as_ref().ravel()
    }
}

impl Hash for dyn BSDF {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id().hash(state);
        self.dyn_hash(state)
    }
}
