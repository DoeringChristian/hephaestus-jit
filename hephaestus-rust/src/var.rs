use std::marker::PhantomData;

use half::f16;
use jit;

#[derive(Clone, Debug)]
pub struct Var<T>(pub(crate) jit::VarRef, pub(crate) PhantomData<T>);

impl<T: jit::AsVarType> From<jit::VarRef> for Var<T> {
    fn from(value: jit::VarRef) -> Self {
        assert_eq!(T::var_ty(), value.ty());
        Self(value, PhantomData)
    }
}

impl<T: jit::AsVarType> jit::Traverse for Var<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>, layout: &mut Vec<usize>) {
        layout.push(0);
        vars.push(self.0.clone());
    }
}
impl<T: jit::AsVarType> jit::Construct for Var<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &mut impl Iterator<Item = usize>,
    ) -> Self {
        assert_eq!(layout.next().unwrap(), 0);
        vars.next().unwrap().into()
    }
}

impl<T> AsRef<Var<T>> for Var<T> {
    fn as_ref(&self) -> &Var<T> {
        &self
    }
}

impl<T> AsRef<jit::VarRef> for Var<T> {
    fn as_ref(&self) -> &jit::VarRef {
        &self.0
    }
}

// Constructors
pub fn array<T: jit::AsVarType>(slice: &[T], device: &jit::Device) -> Var<T> {
    jit::array(slice, device).into()
}
pub fn literal<T: jit::AsVarType>(val: T) -> Var<T> {
    jit::literal(val).into()
}
pub fn sized_literal<T: jit::AsVarType>(val: T, size: usize) -> Var<T> {
    jit::sized_literal(val, size).into()
}
// Index Constructors
pub fn index() -> Var<u32> {
    jit::index().into()
}
pub fn sized_index(size: usize) -> Var<u32> {
    jit::sized_index(size).into()
}
pub fn dyn_index(capacity: usize, size: impl AsRef<Var<u32>>) -> Var<u32> {
    jit::dynamic_index(capacity, &size.as_ref().0).into()
}
// Composite Constructors
pub fn arr<'a, const N: usize, T: jit::AsVarType + 'a>(vars: [impl AsRef<Var<T>>; N]) -> Var<T> {
    // TODO: use threadlocal vec to collect
    let refs = vars
        .into_iter()
        .map(|var| var.as_ref().0.clone())
        .collect::<Vec<_>>();
    jit::arr(&refs).into()
}

pub fn composite<T: jit::AsVarType>() -> CompositeBuilder<T> {
    CompositeBuilder {
        _ty: PhantomData,
        elems: vec![],
    }
}
pub struct CompositeBuilder<T> {
    _ty: PhantomData<T>,
    elems: Vec<jit::VarRef>,
}
impl<'a, T: jit::AsVarType> CompositeBuilder<T> {
    pub fn elem<U: jit::AsVarType>(mut self, elem: impl AsRef<Var<U>>) -> Self {
        self.elems.push(elem.as_ref().0.clone());
        self
    }
    pub fn construct(self) -> Var<T> {
        jit::composite(&self.elems).into()
    }
}

// Extraction
impl<'a, T: jit::AsVarType> Var<T> {
    pub fn extract<U: jit::AsVarType>(&self, elem: usize) -> Var<U> {
        self.0.extract(elem).into()
    }
    pub fn extract_dyn<U: jit::AsVarType>(&self, elem: &Var<u32>) -> Var<U> {
        self.0.extract_dyn(&elem.0).into()
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
            self.0.$op().into()
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
        pub fn $op(&self, other: impl AsRef<Self>) -> Var<$result_type> {
            self.0.$op(&other.as_ref().0).into()
        }
    };
    ($op:ident) => {
        pub fn $op(&self, other: impl AsRef<Self>) -> Self {
            self.0.$op(&other.as_ref().0).into()
        }
    };
}

// Binary Operations
impl<T: jit::AsVarType> Var<T> {
    // Arithmetic
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
    pub fn shr(&self, offset: impl AsRef<Var<i32>>) -> Self {
        self.0.shr(&offset.as_ref().0).into()
    }
    pub fn shl(&self, offset: impl AsRef<Var<i32>>) -> Self {
        self.0.shl(&offset.as_ref().0).into()
    }
}

// Trinary Operations
impl<T: jit::AsVarType> Var<T> {
    pub fn fma(&self, b: impl AsRef<Self>, c: impl AsRef<Self>) -> Self {
        Self(self.0.fma(&b.as_ref().0, &c.as_ref().0), PhantomData)
    }
}
impl<T: jit::AsVarType> Var<T> {
    pub fn select(&self, condition: impl AsRef<Var<bool>>, false_val: impl AsRef<Self>) -> Self {
        condition
            .as_ref()
            .0
            .select(&self.0, &false_val.as_ref().0)
            .into()
    }
}

// Casting
impl<T: jit::AsVarType> Var<T> {
    pub fn cast<U: jit::AsVarType>(&self) -> Var<U> {
        self.0.cast(U::var_ty()).into()
    }
    pub fn bitcast<U: jit::AsVarType>(&self) -> Var<U> {
        self.0.bitcast(U::var_ty()).into()
    }
}

// Gather/Scatter
impl<T: jit::AsVarType> Var<T> {
    pub fn gather(&self, index: impl AsRef<Var<u32>>) -> Self {
        self.0.gather(&index.as_ref().0).into()
    }
    pub fn gather_if(&self, index: impl AsRef<Var<u32>>, condition: impl AsRef<Var<bool>>) -> Self {
        self.0
            .gather_if(&index.as_ref().0, &condition.as_ref().0)
            .into()
    }

    pub fn scatter(&self, dst: impl AsRef<Self>, index: impl AsRef<Var<u32>>) {
        self.0.scatter(&dst.as_ref().0, &index.as_ref().0)
    }
    pub fn scatter_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    ) {
        self.0
            .scatter_if(&dst.as_ref().0, &index.as_ref().0, &condition.as_ref().0)
    }

    pub fn scatter_reduce(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        op: jit::ReduceOp,
    ) {
        self.0
            .scatter_reduce(&dst.as_ref().0, &index.as_ref().0, op)
    }
    pub fn scatter_reduce_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
        op: jit::ReduceOp,
    ) {
        self.0.scatter_reduce_if(
            &dst.as_ref().0,
            &index.as_ref().0,
            &condition.as_ref().0,
            op,
        )
    }

    pub fn scatter_atomic(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        op: jit::ReduceOp,
    ) -> Self {
        self.0
            .scatter_atomic(&dst.as_ref().0, &index.as_ref().0, op)
            .into()
    }
    pub fn scatter_atomic_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
        op: jit::ReduceOp,
    ) -> Self {
        self.0
            .scatter_atomic_if(
                &dst.as_ref().0,
                &index.as_ref().0,
                &condition.as_ref().0,
                op,
            )
            .into()
    }

    pub fn atomic_inc(
        &self,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    ) -> Self {
        self.0
            .atomic_inc(&index.as_ref().0, &condition.as_ref().0)
            .into()
    }
}

pub type Float16 = Var<f16>;
pub type Float32 = Var<f32>;
pub type Float64 = Var<f64>;

pub type Int8 = Var<i8>;
pub type Int16 = Var<i16>;
pub type Int32 = Var<i32>;
pub type Int64 = Var<i64>;

pub type UInt8 = Var<u8>;
pub type UInt16 = Var<u16>;
pub type UInt32 = Var<u32>;
pub type UInt64 = Var<u64>;

#[cfg(test)]
mod test {
    use crate::*;
    use jit::AsVarType;
    #[test]
    fn composite1() {
        let device = vulkan(0);

        #[repr(C)]
        #[derive(AsVarType, Clone, Copy, Debug, PartialEq, Eq)]
        pub struct VecI3 {
            x: i32,
            y: i32,
        }

        #[recorded]
        fn kernel() -> Var<VecI3> {
            let x = sized_literal(1, 10);
            let y = literal(2);
            let v = composite().elem(&x).elem(&y).construct();
            v
        }

        let v = kernel(&device).unwrap().0;
        let reference = (0..10).map(|_| VecI3 { x: 1, y: 2 }).collect::<Vec<_>>();

        assert_eq!(v.to_vec(), reference);
    }
}
