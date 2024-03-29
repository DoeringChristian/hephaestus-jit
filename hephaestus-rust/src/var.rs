use std::marker::PhantomData;

use jit;

pub struct Var<T>(pub(crate) jit::VarRef, pub(crate) PhantomData<T>);

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
    Var::<T>(jit::array(slice, device), PhantomData)
}
pub fn literal<T: jit::AsVarType>(val: T) -> Var<T> {
    Var::<T>(jit::literal(val), PhantomData)
}
pub fn sized_literal<T: jit::AsVarType>(val: T, size: usize) -> Var<T> {
    Var::<T>(jit::sized_literal(val, size), PhantomData)
}
// Index Constructors
pub fn index() -> Var<u32> {
    Var::<u32>(jit::index(), PhantomData)
}
pub fn sized_index(size: usize) -> Var<u32> {
    Var::<u32>(jit::sized_index(size), PhantomData)
}
pub fn dyn_index(capacity: usize, size: impl AsRef<Var<u32>>) -> Var<u32> {
    Var::<u32>(jit::dynamic_index(capacity, &size.as_ref().0), PhantomData)
}
// Composite Constructors
pub fn arr<'a, T: jit::AsVarType + 'a>(vars: impl IntoIterator<Item = &'a Var<T>>) -> Var<T> {
    // TODO: use threadlocal vec to collect
    let refs = vars
        .into_iter()
        .map(|var| var.0.clone())
        .collect::<Vec<_>>();
    Var::<T>(jit::arr(&refs), PhantomData)
}
pub fn vec<'a, T: jit::AsVarType + 'a, A: AsRef<Var<T>>>(
    vars: impl IntoIterator<Item = A>,
) -> Var<T> {
    // TODO: use threadlocal vec to collect
    // Also, figure out better collection method
    let refs = vars
        .into_iter()
        .map(|var| var.as_ref().0.clone())
        .collect::<Vec<_>>();
    Var::<T>(jit::vec(&refs), PhantomData)
}

pub fn mat<'a, T: jit::AsVarType + 'a>(columns: impl IntoIterator<Item = &'a Var<T>>) -> Var<T> {
    // TODO: use threadlocal vec to collect
    let refs = columns
        .into_iter()
        .map(|var| var.0.clone())
        .collect::<Vec<_>>();
    Var::<T>(jit::mat(&refs), PhantomData)
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
        pub fn $op(&self, other: impl AsRef<Self>) -> Var<$result_type> {
            Var::<$result_type>(self.0.$op(&other.as_ref().0), PhantomData)
        }
    };
    ($op:ident) => {
        pub fn $op(&self, other: impl AsRef<Self>) -> Self {
            Self(self.0.$op(&other.as_ref().0), PhantomData)
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
    pub fn shr(&self, offset: impl AsRef<Var<i32>>) -> Self {
        Self(self.0.shr(&offset.as_ref().0), PhantomData)
    }
    pub fn shl(&self, offset: impl AsRef<Var<i32>>) -> Self {
        Self(self.0.shl(&offset.as_ref().0), PhantomData)
    }
}

// Trinary Operations
impl<T: jit::AsVarType> Var<T> {
    pub fn fma(&self, b: impl AsRef<Self>, c: impl AsRef<Self>) -> Self {
        Self(self.0.fma(&b.as_ref().0, &c.as_ref().0), PhantomData)
    }
}
impl Var<bool> {
    pub fn select<T: jit::AsVarType>(
        &self,
        true_val: impl AsRef<Var<T>>,
        false_val: impl AsRef<Var<T>>,
    ) -> Var<T> {
        Var::<T>(
            self.0.select(&true_val.as_ref().0, &false_val.as_ref().0),
            PhantomData,
        )
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
    pub fn gather(&self, index: impl AsRef<Var<u32>>) -> Self {
        Self(self.0.gather(&index.as_ref().0), PhantomData)
    }
    pub fn gather_if(&self, index: impl AsRef<Var<u32>>, condition: impl AsRef<Var<bool>>) -> Self {
        Self(
            self.0.gather_if(&index.as_ref().0, &condition.as_ref().0),
            PhantomData,
        )
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
        Self(
            self.0
                .scatter_atomic(&dst.as_ref().0, &index.as_ref().0, op),
            PhantomData,
        )
    }
    pub fn scatter_atomic_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
        op: jit::ReduceOp,
    ) -> Self {
        Self(
            self.0.scatter_atomic_if(
                &dst.as_ref().0,
                &index.as_ref().0,
                &condition.as_ref().0,
                op,
            ),
            PhantomData,
        )
    }

    pub fn atomic_inc(
        &self,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    ) -> Self {
        Self(
            self.0.atomic_inc(&index.as_ref().0, &condition.as_ref().0),
            PhantomData,
        )
    }
}

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
