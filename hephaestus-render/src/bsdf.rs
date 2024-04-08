use std::any::Any;
use std::hash::{DefaultHasher, Hash, Hasher};

use hephaestus as hep;

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

    fn hash(&self, state: &mut dyn std::hash::Hasher) {
        let mut hasher = DefaultHasher::new();
        self.type_id().hash(&mut hasher);
        state.write_u64(hasher.finish());
        self.as_ref().hash(state);
    }
}
