use std::marker::PhantomData;
use std::ops::Deref;

use half::f16;
use jit;

use crate::Scatter;

#[derive(Clone, Debug)]
pub struct Var<T>(pub(crate) jit::VarRef, pub(crate) PhantomData<T>);

impl<T: jit::AsVarType> From<jit::VarRef> for Var<T> {
    fn from(value: jit::VarRef) -> Self {
        assert_eq!(T::var_ty(), value.ty());
        Self(value, PhantomData)
    }
}
impl<T: jit::AsVarType> From<&jit::VarRef> for Var<T> {
    fn from(value: &jit::VarRef) -> Self {
        assert_eq!(T::var_ty(), value.ty());
        Self(value.clone(), PhantomData)
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
impl<T> Deref for Var<T> {
    type Target = jit::VarRef;

    fn deref(&self) -> &Self::Target {
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

macro_rules! uop_trait {
    ($op:ident for $($types:ident),*) => {
        uop_trait!($op -> (Self) for $($types),*);
    };
    ($op:ident -> ($ret_type:ty) for $($types:ty),*) => {
        paste::paste! {
            pub trait [<$op:camel>] {
                type Return;
                fn $op(&self) -> Self::Return;
            }
            $(
                impl [<$op:camel>] for Var<$types> {
                    type Return = $ret_type;
                    fn $op(&self) -> Self::Return{
                        self.0.$op().into()
                    }
                }
            )*
            impl<T: jit::AsVarType> Var<T>
            where
                Var<T>: [<$op:camel>],
            {
                pub fn $op(&self) -> <Self as [<$op:camel>]>::Return {
                    <Self as [<$op:camel>]>::$op(self)
                }
            }
        }
    };
}

// Unary Operations
uop_trait!(neg for bool, i8, i16, i32, i64, f16, f32, f64);
uop_trait!(sqrt for f16, f32, f64);
uop_trait!(abs for i8, i16, i32, i64, f16, f32, f64);
uop_trait!(sin for f16, f32, f64);
uop_trait!(cos for f16, f32, f64);
uop_trait!(exp2 for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
uop_trait!(log2 for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);

macro_rules! bop_trait {
    ($op:ident for $($types:ident),*) => {
        bop_trait!($op (self, jit::VarRef) -> (Self) for $($types),*);
    };
    ($op:ident -> ($ret_type:ty) for $($types:ty),*) => {
        bop_trait!($op (self, jit::VarRef) -> ($ret_type) for $($types),*);
    };
    ($op:ident (self, $rhs:ty) -> ($ret_type:ty) for $($types:ty),*) => {
        paste::paste! {
            pub trait [<$op:camel>] {
                type Return;
                type Rhs;
                fn $op(&self, other: impl AsRef<Self::Rhs>) -> Self::Return;
            }
            $(
                impl [<$op:camel>] for Var<$types> {
                    type Return = $ret_type;
                    type Rhs = $rhs;
                    fn $op(&self, other: impl AsRef<$rhs>) -> $ret_type {
                        self.0.$op(other.as_ref()).into()
                    }
                }
            )*
            impl<T: jit::AsVarType> Var<T>
            where
                Var<T>: [<$op:camel>],
            {
                pub fn $op(&self, other: impl AsRef<<Self as [<$op:camel>]>::Rhs>) -> <Self as [<$op:camel>]>::Return {
                    <Self as [<$op:camel>]>::$op(self, other)
                }
            }
        }
    };
}

// Arithmetic
bop_trait!(sub for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(add for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(mul for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(div for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(modulus for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(min for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(max for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);

macro_rules! std_bop {
    ($op:ident) => {
        paste::paste! {
            impl<T: jit::AsVarType, Rhs: AsRef<<Var<T> as [<$op:camel>]>::Rhs>> std::ops::[<$op:camel>]<Rhs> for Var<T>
            where
                Var<T>: [<$op:camel>],
            {
                type Output = <Var<T> as [<$op:camel>]>::Return;

                fn $op(self, rhs: Rhs) -> Self::Output {
                    [<$op:camel>]::$op(&self, rhs)
                }
            }
        }
    };
}

std_bop!(add);
std_bop!(sub);
std_bop!(mul);
std_bop!(div);

// Bitwise
bop_trait!(and for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(or for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(xor for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);

// Comparisons
bop_trait!(eq (self, Self) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(neq (self, Self) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(lt (self, Self) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(le (self, Self) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(gt (self, Self) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(ge (self, Self) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);

// Shift
bop_trait!(shr (self, Var<i32>) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);
bop_trait!(shl (self, Var<i32>) -> (Var<bool>) for bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);

macro_rules! top_trait {
    ($op:ident for $($types:ident),*) => {
        top_trait!($op (self, Self, Self) -> (Self) for $($types),*);
    };
    ($op:ident -> ($ret_type:ty) for $($types:ty),*) => {
        top_trait!($op (self, Self, Self) -> ($ret_type) for $($types),*);
    };
    ($op:ident (self, $b:ty, $c:ty) -> ($ret_type:ty) for $($types:ty),*) => {
        paste::paste! {
            pub trait [<$op:camel>] {
                type Return;
                fn $op(&self, b: impl AsRef<$b>, c: impl AsRef<$c>) -> Self::Return;
            }
            $(
                impl [<$op:camel>] for Var<$types> {
                    type Return = $ret_type;
                    fn $op(&self, b: impl AsRef<$b>, c: impl AsRef<$c>) -> $ret_type {
                        self.0.$op(&b.as_ref(), &c.as_ref()).into()
                    }
                }
            )*
            impl<T: jit::AsVarType> Var<T>
            where
                Var<T>: [<$op:camel>],
            {
                pub fn $op(&self, b: impl AsRef<$b>, c: impl AsRef<$c>) -> <Self as [<$op:camel>]>::Return {
                    <Self as [<$op:camel>]>::$op(self, b, c)
                }
            }
        }
    };
}

// Trinary Operations
top_trait!(fma for i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64);

impl<T: jit::AsVarType> Var<T> {
    // Select is implemented for all variables
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

impl<T> Var<T>
where
    Var<T>: Scatter,
{
    pub fn scatter(&self, dst: impl AsRef<Self>, index: impl AsRef<Var<u32>>) {
        Scatter::scatter(self, dst, index)
    }
    pub fn scatter_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    ) {
        Scatter::scatter_if(self, dst, index, condition)
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
