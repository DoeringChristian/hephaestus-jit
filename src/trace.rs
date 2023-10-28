#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(pub(crate) usize);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum VarType {
    // Primitive Types (might move out)
    #[default]
    Void,
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    // Array,
}
impl VarType {
    pub fn size(&self) -> usize {
        match self {
            VarType::Void => 0,
            VarType::Bool => 1,
            VarType::I8 => 1,
            VarType::U8 => 1,
            VarType::I16 => 2,
            VarType::U16 => 2,
            VarType::I32 => 4,
            VarType::U32 => 4,
            VarType::I64 => 8,
            VarType::U64 => 8,
            VarType::F32 => 4,
            VarType::F64 => 8,
            // VarType::Array => 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Var {
    pub(crate) ty: VarType,
    pub(crate) op: Op,
    // pub(crate) deps: (usize, usize),
}

#[derive(Clone, Default, Debug)]
pub enum Op {
    #[default]
    Nop,
    LoadArray,
    Add {
        lhs: VarId,
        rhs: VarId,
    },
    Scatter {
        dst: VarId,
        src: VarId,
        idx: VarId,
    },
    Gather {
        src: VarId,
        idx: VarId,
    },
    Index,
    Const {
        data: u64,
    },
}

#[derive(Debug, Default)]
pub struct Trace {
    pub(crate) arrays: Vec<VarId>,
    pub(crate) vars: Vec<Var>,
    pub(crate) size: usize,
}

impl Trace {
    // pub fn push_var(&mut self, mut var: Var, size: usize, deps: &[VarId]) -> VarId {
    pub fn push_var(&mut self, mut var: Var, size: usize) -> VarId {
        let id = VarId(self.vars.len());
        self.size = self.size.max(size);

        // // Add dependencies
        // let start = self.deps.len();
        // self.deps.extend_from_slice(deps);
        // let stop = self.deps.len();

        // var.deps = (start, stop);
        self.vars.push(var);
        id
    }
    pub fn push_array(&mut self, var: Var) -> VarId {
        let id = self.push_var(var, 0);
        self.arrays.push(id);
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
}
