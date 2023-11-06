#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(pub(crate) usize);
use crate::op::Op;

use crate::vartype::VarType;

#[derive(Clone, Debug, Default)]
pub struct Var {
    pub(crate) ty: VarType,
    pub(crate) op: Op,
    pub(crate) deps: (usize, usize),
}

#[derive(Debug, Default)]
pub struct IR {
    pub(crate) vars: Vec<Var>,
    pub(crate) size: usize,
    pub(crate) deps: Vec<VarId>,
}

impl IR {
    // pub fn push_var(&mut self, mut var: Var, size: usize, deps: &[VarId]) -> VarId {
    pub fn push_var(&mut self, mut var: Var, deps: &[VarId]) -> VarId {
        let id = VarId(self.vars.len());

        // Add dependencies
        let start = self.deps.len();
        self.deps.extend_from_slice(deps);
        let stop = self.deps.len();

        // var.deps = (start, stop);
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
}
