use std::cell::RefCell;

use crate::backend::{self, Array, Device, Parameters};
use crate::trace::{Op, OpId, Trace, Var, VarId, VarType};

#[derive(Default, Debug)]
struct InternalTrace {
    pub(crate) trace: Trace,
    pub(crate) arrays: Vec<Array>,
}

#[derive(Default, Debug)]
pub struct Kernel(RefCell<InternalTrace>);

impl Kernel {
    pub fn push_op(&self, op: Op) -> OpId {
        self.0.borrow_mut().trace.push_op(op)
    }
    pub fn push_var(&self, var: Var) -> VarId {
        self.0.borrow_mut().trace.push_var(var)
    }
    pub fn push_array(&self, var: Var) -> VarId {
        self.0.borrow_mut().trace.push_array(var)
    }
}

#[derive(Clone, Copy)]
pub struct VarRef<'a> {
    id: VarId,
    r: &'a Kernel,
}

impl<'a> VarRef<'a> {
    pub fn ty(&self) -> VarType {
        self.r.0.borrow_mut().trace.var_ty(self.id).clone()
    }
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.ty(), other.ty());
        let ty = self.ty();
        let dst = self.r.0.borrow_mut().trace.push_var(Var {
            ty,
            ..Default::default()
        });
        self.r.push_op(Op::Add {
            dst,
            lhs: self.id,
            rhs: other.id,
        });
        Self { id: dst, r: self.r }
    }
    pub fn scatter(&self, target: &Self, idx: &Self) {
        self.r.push_op(Op::Scatter {
            src: self.id,
            dst: target.id,
            idx: idx.id,
        });
    }
    pub fn gather(&self, idx: &Self) -> Self {
        let ty = self.ty();
        let dst = self.r.push_var(Var {
            ty,
            ..Default::default()
        });
        self.r.push_op(Op::Gather {
            src: self.id,
            dst,
            idx: idx.id,
        });
        return Self { id: dst, r: self.r };
    }
}

impl Kernel {
    pub fn index<'a>(&'a self, size: usize) -> VarRef<'a> {
        let id = self.push_var(Var {
            ty: VarType::U32,
            size,
        });
        self.push_op(Op::Index { dst: id });
        VarRef { id, r: self }
    }
    pub fn array<'a>(&'a self, array: &Array) -> VarRef<'a> {
        self.0.borrow_mut().arrays.push(array.clone());
        let id = self.push_array(Var {
            ty: array.ty(),
            ..Default::default()
        });
        VarRef { id, r: self }
    }
    pub fn launch(&self, device: &Device) -> backend::Result<()> {
        let size = self.0.borrow_mut().trace.size;
        device.execute_trace(
            &self.0.borrow().trace,
            Parameters {
                size: size as _,
                arrays: self.0.borrow().arrays.clone(),
            },
        )
    }
}

#[cfg(test)]
mod test {

    use crate::backend::Device;
    use crate::trace::VarType;

    use super::Kernel;

    #[test]
    fn index() {
        let device = Device::cuda(0).unwrap();
        let output = device.create_array(10, VarType::U32).unwrap();

        let k = Kernel::default();

        {
            let output = k.array(&output);
            let idx = k.index(10);

            idx.scatter(&output, &idx);
        }

        k.launch(&device).unwrap();

        dbg!(output.to_host().unwrap());
    }
}
