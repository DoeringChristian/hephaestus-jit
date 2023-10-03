#[derive(Clone, Copy, Debug)]
pub struct VarId(pub(crate) usize);

#[derive(Clone, Debug)]
pub enum VarType {
    // Primitive Types (might move out)
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
    Array(Box<VarType>),
}

#[derive(Clone, Debug)]
pub struct Var {
    pub(crate) ty: VarType,
}

#[derive(Clone, Copy, Debug)]
pub struct OpId(pub(crate) usize);

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add { dst: VarId, lhs: VarId, rhs: VarId },
    Scatter { dst: VarId, src: VarId, idx: VarId },
    Gather { dst: VarId, src: VarId, idx: VarId },
}

#[derive(Debug)]
pub struct Trace {
    pub(crate) ops: Vec<Op>,
    pub(crate) vars: Vec<Var>,
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
