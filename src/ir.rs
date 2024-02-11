use crate::op::KernelOp;
use crate::tr::ScopeId;
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
    pub(crate) scope: ScopeId,
}
impl Default for Var {
    fn default() -> Self {
        Self {
            ty: vartype::void(),
            op: Default::default(),
            deps: Default::default(),
            data: Default::default(),
            scope: Default::default(),
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
    pub(crate) hash: Mutex<Option<u64>>,
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
    pub fn scope_sort(&mut self) {
        self.invalidate();

        // Calculate sorting indices by sorting by scope
        let mut indices = (0..self.vars.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| self.vars[i].scope);

        // Change dependencies
        for id in self.deps.iter_mut() {
            *id = VarId(indices[id.0]);
        }

        // Scatter variables at their sorted location
        let mut vars = Vec::with_capacity(self.vars.len());

        // SAFETY: this is safe because [indices] represents a valid permutation
        // and we initialize the array just after this.
        unsafe { vars.set_len(self.vars.len()) };

        for (i, var) in indices.into_iter().zip(&self.vars) {
            vars[i] = *var;
        }
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
    pub fn internal_hash(&self) -> u64 {
        *self.hash.lock().unwrap().get_or_insert_with(|| {
            let mut hasher = DefaultHasher::default();
            self.vars.hash(&mut hasher);
            self.deps.hash(&mut hasher);
            self.n_buffers.hash(&mut hasher);
            self.n_textures.hash(&mut hasher);
            self.n_accels.hash(&mut hasher);
            hasher.finish()
        })
    }
    /// Invalidate Hash
    pub fn invalidate(&mut self) {
        *self.hash.lock().unwrap() = None;
    }
}
