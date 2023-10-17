use std::cell::RefCell;

use crate::trace::{Op, Trace, Var, VarId, VarType};

#[derive(Default, Debug)]
pub struct Tracer {
    trace: RefCell<Trace>,
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
        let dst = self.r.trace.borrow_mut().push_var(Var { ty });
        self.r.trace.borrow_mut().push_op(Op::Add {
            dst,
            lhs: self.id,
            rhs: other.id,
        });
        Self { id: dst, r: self.r }
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
    pub fn array<'a>(&'a self, ident: impl AsRef<str>) -> VarRef<'a> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::Tracer;

    #[test]
    fn index() {
        let mut t = Tracer::default();

        let idx = t.index();

        let idx2 = idx.add(&idx);

        let arr = t.array("test");

        dbg!(t);
    }
}
