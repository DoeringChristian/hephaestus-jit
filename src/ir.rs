#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(pub(crate) usize);

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::op::Op;

use crate::vartype::VarType;

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct Var {
    pub(crate) ty: VarType,
    pub(crate) op: Op,
    pub(crate) deps: (usize, usize),
    pub(crate) data: u64,
}

#[derive(Debug, Default)]
pub struct IR {
    pub(crate) vars: Vec<Var>,
    pub(crate) deps: Vec<VarId>,
    pub(crate) n_buffers: usize,
    pub(crate) hash: Option<u64>,
}

impl IR {
    // pub fn push_var(&mut self, mut var: Var, size: usize, deps: &[VarId]) -> VarId {
    pub fn push_var(&mut self, mut var: Var, deps: impl IntoIterator<Item = VarId>) -> VarId {
        self.invalidate();

        let id = VarId(self.vars.len());

        // Add dependencies
        let start = self.deps.len();
        // self.deps.extend_from_slice(deps);
        self.deps.extend(deps);
        let stop = self.deps.len();

        var.deps = (start, stop);
        self.vars.push(var);
        id
    }
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_ty(&self, id: VarId) -> &VarType {
        &self.var(id).ty
    }
    pub fn var_ids(&self) -> impl Iterator<Item = VarId> {
        (0..self.vars.len()).map(|i| VarId(i))
    }
    pub fn deps(&self, id: VarId) -> &[VarId] {
        let (deps_start, deps_end) = self.var(id).deps;
        &self.deps[deps_start..deps_end]
    }
    pub fn hash(&mut self) -> u64 {
        *self.hash.get_or_insert_with(|| {
            let mut hasher = DefaultHasher::default();
            self.vars.hash(&mut hasher);
            self.deps.hash(&mut hasher);
            self.n_buffers.hash(&mut hasher);
            hasher.finish()
        })
    }
    /// Invalidate Hash
    pub fn invalidate(&mut self) {
        self.hash = None;
    }
}
