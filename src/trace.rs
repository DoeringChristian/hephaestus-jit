#[derive(Clone, Copy, Debug)]
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
    F16,
    F32,
    F64,
    Array,
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
            VarType::F16 => 2,
            VarType::F32 => 4,
            VarType::F64 => 8,
            VarType::Array => 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Var {
    pub(crate) ty: VarType,
    // External Array, Texture or Acceleration structure (not
    // quite sure how to handle this)
    pub(crate) external: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct OpId(pub(crate) usize);

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add { dst: VarId, lhs: VarId, rhs: VarId },
    Scatter { dst: VarId, src: VarId, idx: VarId },
    Gather { dst: VarId, src: VarId, idx: VarId },
}

#[derive(Debug, Default)]
pub struct Trace {
    pub(crate) ops: Vec<Op>,
    pub(crate) vars: Vec<Var>,
    pub(crate) n_externals: usize,
}

impl Trace {
    pub fn push_var(&mut self, var: Var) -> VarId {
        let id = VarId(self.vars.len());
        self.vars.push(var);
        id
    }
    pub fn push_op(&mut self, op: Op) -> OpId {
        let id = OpId(self.ops.len());
        self.ops.push(op);
        id
    }
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_ty(&self, id: VarId) -> &VarType {
        &self.var(id).ty
    }
    pub fn op(&self, id: OpId) -> &Op {
        &self.ops[id.0]
    }
}
