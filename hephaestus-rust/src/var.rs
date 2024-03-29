use std::marker::PhantomData;

use hephaestus_jit as jit;

pub struct Var<T>(jit::VarRef, PhantomData<T>);

impl<T> jit::Traverse for Var<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>, layout: &mut Vec<usize>) {
        layout.push(0);
        vars.push(self.0.clone());
    }
}

impl<T> AsRef<jit::VarRef> for Var<T> {
    fn as_ref(&self) -> &jit::VarRef {
        &self.0
    }
}

// Constructors
impl<T: jit::AsVarType> Var<T> {
    pub fn array(slice: &[T], device: &jit::Device) -> Self {
        Self(jit::array(slice, device), PhantomData)
    }
    pub fn literal(val: T) -> Self {
        Self(jit::literal(val), PhantomData)
    }
    pub fn sized_literal(val: T, size: usize) -> Self {
        Self(jit::sized_literal(val, size), PhantomData)
    }
}
// Index Constructors
impl Var<u32> {
    pub fn index() -> Self {
        Self(jit::index(), PhantomData)
    }
    pub fn sized_index(size: usize) -> Self {
        Self(jit::sized_index(size), PhantomData)
    }
    pub fn dynamic_index(capacity: usize, size: &Var<u32>) -> Self {
        Self(jit::dynamic_index(capacity, &size.0), PhantomData)
    }
}
// Composite Constructors
impl<T: jit::AsVarType> Var<T> {
    pub fn arr(vars: &[Var<T>]) -> Self {
        // TODO: use threadlocal vec to collect
        let refs = vars.into_iter().map(|var| &var.0).collect::<Vec<_>>();
        Self(jit::arr(&refs), PhantomData)
    }
    pub fn vec(vars: &[Var<T>]) -> Self {
        // TODO: use threadlocal vec to collect
        let refs = vars.into_iter().map(|var| &var.0).collect::<Vec<_>>();
        Self(jit::vec(&refs), PhantomData)
    }
    pub fn mat(columns: &[Var<T>]) -> Self {
        // TODO: use threadlocal vec to collect
        let refs = columns.into_iter().map(|var| &var.0).collect::<Vec<_>>();
        Self(jit::mat(&refs), PhantomData)
    }
}

impl<T: jit::AsVarType> Var<T> {
    pub fn composite<'a>() -> CompositeBuilder<'a, T> {
        CompositeBuilder {
            _ty: PhantomData,
            elems: vec![],
        }
    }
}
pub struct CompositeBuilder<'a, T> {
    _ty: PhantomData<T>,
    elems: Vec<&'a jit::VarRef>,
}
impl<'a, T: jit::AsVarType> CompositeBuilder<'a, T> {
    pub fn elem<U: jit::AsVarType>(mut self, elem: &'a Var<U>) -> Self {
        self.elems.push(&elem.0);
        self
    }
    pub fn construct(self) -> Var<T> {
        Var::<T>(jit::composite(&self.elems), PhantomData)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use jit::AsVarType;
    #[test]
    fn composite() {
        #[repr(C)]
        #[derive(AsVarType, Clone, Copy)]
        pub struct VecI3 {
            x: i32,
            y: i32,
        }
        let x = Var::sized_literal(1, 10);
        let y = Var::literal(1);
        let v = Var::<VecI3>::composite().elem(&x).elem(&y).construct();
    }
}
