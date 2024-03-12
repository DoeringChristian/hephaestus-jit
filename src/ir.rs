use itertools::Itertools;

use crate::op::KernelOp;
// use crate::tr::ScopeId;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use crate::vartype;
use crate::vartype::VarType;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(pub(crate) usize);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Var {
    pub(crate) ty: &'static VarType,
    pub(crate) op: KernelOp,
    pub(crate) deps: (usize, usize),
    pub(crate) data: u64,
    // pub(crate) scope: ScopeId,
}
impl Default for Var {
    fn default() -> Self {
        Self {
            ty: vartype::void(),
            op: Default::default(),
            deps: Default::default(),
            data: Default::default(),
            // scope: Default::default(),
        }
    }
}
impl Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("ty", &self.ty)
            .field("op", &self.op)
            .field("deps", &self.deps)
            .field("data", &self.data)
            // .field("scope", &self.scope)
            .finish()
    }
}
impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ty.hash(state);
        self.op.hash(state);
        self.deps.hash(state);
        self.data.hash(state);
    }
}

#[derive(Debug, Default)]
pub struct IR {
    pub(crate) vars: Vec<Var>,
    pub(crate) deps: Vec<VarId>,
    pub(crate) n_buffers: usize,
    pub(crate) n_textures: usize,
    pub(crate) n_accels: usize,
}
// Implement hash for IR using it's cached internal hash
impl Hash for IR {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vars.hash(state);
        self.deps.hash(state);
        self.n_buffers.hash(state);
        self.n_textures.hash(state);
        self.n_accels.hash(state);
    }
}

impl IR {
    // pub fn push_var(&mut self, mut var: Var, size: usize, deps: &[VarId]) -> VarId {
    pub fn push_var(&mut self, mut var: Var, deps: impl IntoIterator<Item = VarId>) -> VarId {
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
    pub fn var_ty(&self, id: VarId) -> &'static VarType {
        self.var(id).ty
    }
    pub fn var_ids(&self) -> impl Iterator<Item = VarId> {
        (0..self.vars.len()).map(|i| VarId(i))
    }
    pub fn deps(&self, id: VarId) -> &[VarId] {
        let (deps_start, deps_end) = self.var(id).deps;
        &self.deps[deps_start..deps_end]
    }
    pub fn clear(&mut self) {
        self.vars.clear();
        self.deps.clear();
        self.n_buffers = 0;
        self.n_textures = 0;
        self.n_accels = 0;
    }
}
