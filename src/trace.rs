#[derive(Clone, Copy, Debug)]
pub struct VarId(usize);

#[derive(Clone, Copy, Debug)]
pub enum VarType {
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
}

pub struct Var {
    ty: VarType,
}

#[derive(Clone, Copy, Debug)]
pub struct OpId(usize);

pub enum Op {
    Add { lhs: VarId, rhs: VarId, dst: VarId },
}

pub struct Trace {
    ops: Vec<Op>,
    vars: Vec<Var>,
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
}
