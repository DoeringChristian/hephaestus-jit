use std::marker::PhantomData;

use hephaestus_jit as jit;

pub struct Var<T>(jit::VarRef, PhantomData<T>);

impl<T> jit::Traverse for Var<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>, layout: &mut Vec<usize>) {
        layout.push(0);
        vars.push(self.0.clone());
    }
}
impl<T> jit::Construct for Var<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &mut impl Iterator<Item = usize>,
    ) -> Self {
        assert_eq!(layout.next().unwrap(), 0);
        Self(vars.next().unwrap(), PhantomData)
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

// Extraction
impl<'a, T: jit::AsVarType> Var<T> {
    pub fn extract<U: jit::AsVarType>(&self, elem: usize) -> Var<U> {
        Var::<U>(self.0.extract(elem), PhantomData)
    }
    pub fn extract_dyn<U: jit::AsVarType>(&self, elem: &Var<u32>) -> Var<U> {
        Var::<U>(self.0.extract_dyn(&elem.0), PhantomData)
    }
}

// To Host functions
impl<T: jit::AsVarType> Var<T> {
    pub fn to_vec(&self) -> Vec<T> {
        self.0.to_vec(..)
    }
    pub fn to_vec_range(&self, range: impl std::ops::RangeBounds<usize>) -> Vec<T> {
        self.0.to_vec(range)
    }
    pub fn item(&self) -> T {
        self.0.item()
    }
}

// Utility functions
impl<T: jit::AsVarType> Var<T> {
    pub fn id(&self) -> jit::VarId {
        self.0.id()
    }
    pub fn schedule(&self) {
        self.0.schedule()
    }
    pub fn is_evaluated(&self) -> bool {
        self.0.is_evaluated()
    }
    pub fn rc(&self) -> usize {
        self.0.rc()
    }
    pub fn ty(&self) -> &'static jit::VarType {
        self.0.ty()
    }
}

macro_rules! uop {
    ($op:ident) => {
        pub fn $op(&self) -> Self {
            Self(self.0.$op(), PhantomData)
        }
    };
}

// Unary Operations
impl<T: jit::AsVarType> Var<T> {
    uop!(neg);
    uop!(sqrt);
    uop!(abs);
    uop!(sin);
    uop!(cos);
    uop!(exp2);
    uop!(log2);
}

macro_rules! bop {
    ($op:ident -> $result_type:ident) => {
        pub fn $op(&self, other: &Self) -> Var<$result_type> {
            Var::<$result_type>(self.0.$op(&other.0), PhantomData)
        }
    };
    ($op:ident) => {
        pub fn $op(&self, other: &Self) -> Self {
            Self(self.0.$op(&other.0), PhantomData)
        }
    };
}

// Binary Operations
impl<T: jit::AsVarType> Var<T> {
    // Binary operations returing the same type
    bop!(add);
    bop!(sub);
    bop!(mul);
    bop!(div);
    bop!(modulus);
    bop!(min);
    bop!(max);

    // Bitwise
    bop!(and);
    bop!(or);
    bop!(xor);

    // Comparisons
    bop!(eq -> bool);
    bop!(neq -> bool);
    bop!(lt -> bool);
    bop!(le -> bool);
    bop!(gt -> bool);
    bop!(ge -> bool);

    // Shift
    pub fn shr(&self, offset: &Var<i32>) -> Self {
        Self(self.0.shr(&offset.0), PhantomData)
    }
    pub fn shl(&self, offset: &Var<i32>) -> Self {
        Self(self.0.shl(&offset.0), PhantomData)
    }
}

// Trinary Operations
impl<T: jit::AsVarType> Var<T> {
    pub fn fma(&self, b: &Self, c: &Self) -> Self {
        Self(self.0.fma(&b.0, &c.0), PhantomData)
    }
}
impl Var<bool> {
    pub fn select<T: jit::AsVarType>(&self, true_val: &Var<T>, false_val: &Var<T>) -> Var<T> {
        Var::<T>(self.0.select(&true_val.0, &false_val.0), PhantomData)
    }
}

// Casting
impl<T: jit::AsVarType> Var<T> {
    pub fn cast<U: jit::AsVarType>(&self) -> Var<U> {
        Var::<U>(self.0.cast(U::var_ty()), PhantomData)
    }
    pub fn bitcast<U: jit::AsVarType>(&self) -> Var<U> {
        Var::<U>(self.0.bitcast(U::var_ty()), PhantomData)
    }
}

// Gather/Scatter
impl<T: jit::AsVarType> Var<T> {
    pub fn gather(&self, index: &Var<u32>) -> Self {
        Self(self.0.gather(&index.0), PhantomData)
    }
    pub fn gather_if(&self, index: &Var<u32>, condition: &Var<bool>) -> Self {
        Self(self.0.gather_if(&index.0, &condition.0), PhantomData)
    }

    pub fn scatter(&self, dst: &Self, index: &Var<u32>) {
        self.0.scatter(&dst.0, &index.0)
    }
    pub fn scatter_if(&self, dst: &Self, index: &Var<u32>, condition: &Var<bool>) {
        self.0.scatter_if(&dst.0, &index.0, &condition.0)
    }

    pub fn scatter_reduce(&self, dst: &Self, index: &Var<u32>, op: jit::ReduceOp) {
        self.0.scatter_reduce(&dst.0, &index.0, op)
    }
    pub fn scatter_reduce_if(
        &self,
        dst: &Self,
        index: &Var<u32>,
        condition: &Var<bool>,
        op: jit::ReduceOp,
    ) {
        self.0.scatter_reduce_if(&dst.0, &index.0, &condition.0, op)
    }

    pub fn scatter_atomic(&self, dst: &Self, index: &Var<u32>, op: jit::ReduceOp) -> Self {
        Self(self.0.scatter_atomic(&dst.0, &index.0, op), PhantomData)
    }
    pub fn scatter_atomic_if(
        &self,
        dst: &Self,
        index: &Var<u32>,
        condition: &Var<bool>,
        op: jit::ReduceOp,
    ) -> Self {
        Self(
            self.0.scatter_atomic_if(&dst.0, &index.0, &condition.0, op),
            PhantomData,
        )
    }

    pub fn atomic_inc(&self, index: &Var<u32>, condition: &Var<bool>) -> Self {
        Self(self.0.atomic_inc(&index.0, &condition.0), PhantomData)
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use jit::AsVarType;
    #[test]
    fn composite() {
        let device = vulkan(0);

        #[repr(C)]
        #[derive(AsVarType, Clone, Copy, Debug, PartialEq, Eq)]
        pub struct VecI3 {
            x: i32,
            y: i32,
        }

        #[recorded]
        fn kernel() -> Var<VecI3> {
            let x = Var::sized_literal(1, 10);
            let y = Var::literal(2);
            let v = Var::<VecI3>::composite().elem(&x).elem(&y).construct();
            v
        }

        let v = kernel(&device).unwrap().0;
        let reference = (0..10).map(|_| VecI3 { x: 1, y: 2 }).collect::<Vec<_>>();

        assert_eq!(v.to_vec(), reference);
    }
}
