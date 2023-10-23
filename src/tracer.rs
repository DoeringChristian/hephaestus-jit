use std::cell::RefCell;

use crate::backend::Array;
use crate::trace::{AsVarType, Op, Trace, Var, VarId, VarType};

#[derive(Default, Debug)]
pub struct Tracer {
    pub(crate) trace: RefCell<Trace>,
}

#[derive(Clone, Copy)]
pub struct VarRef<'a> {
    id: VarId,
    r: &'a Tracer,
}

impl<'a> VarRef<'a> {
    pub fn ty(&self) -> VarType {
        self.r.trace.borrow_mut().var_ty(self.id).clone()
    }
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.ty(), other.ty());
        let ty = self.ty();
        let dst = self.r.trace.borrow_mut().push_var(Var {
            ty,
            ..Default::default()
        });
        self.r.trace.borrow_mut().push_op(Op::Add {
            dst,
            lhs: self.id,
            rhs: other.id,
        });
        Self { id: dst, r: self.r }
    }
    pub fn scatter(&self, target: &Self, idx: &Self) {
        self.r.trace.borrow_mut().push_op(Op::Scatter {
            src: self.id,
            dst: target.id,
            idx: idx.id,
        });
    }
    pub fn gather(&self, idx: &Self) -> Self {
        let ty = self.ty();
        let dst = self.r.trace.borrow_mut().push_var(Var {
            ty,
            ..Default::default()
        });
        self.r.trace.borrow_mut().push_op(Op::Gather {
            src: self.id,
            dst,
            idx: idx.id,
        });
        return Self { id: dst, r: self.r };
    }
}

impl Tracer {
    pub fn index<'a>(&'a self) -> VarRef<'a> {
        let id = self.trace.borrow_mut().push_var(Var {
            ty: VarType::U32,
            ..Default::default()
        });
        self.trace.borrow_mut().push_op(Op::Index { dst: id });
        VarRef { id, r: self }
    }
    pub fn array<'a, T: bytemuck::Pod + AsVarType>(&'a self, array: &Array<T>) -> VarRef<'a> {
        let id = self
            .trace
            .borrow_mut()
            .push_array_var(Var { ty: T::var_ty() });
        VarRef { id, r: self }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::backend::{Device, Parameters};
    use crate::trace::VarType;

    use super::Tracer;

    #[test]
    fn index() {
        let t = Tracer::default();

        let output = t.array(VarType::U32);
        let idx = t.index();

        idx.scatter(&output, &idx);

        dbg!(&t);

        let device = Device::cuda(0).unwrap();
        let output = device.create_array(10 * 4).unwrap();

        device
            .execute_trace(
                &t.trace.borrow(),
                Parameters {
                    size: 10,
                    arrays: vec![output.clone()],
                },
            )
            .unwrap();

        dbg!(output.to_host().unwrap());
    }
}
