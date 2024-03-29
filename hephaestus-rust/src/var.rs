use std::marker::PhantomData;

use hephaestus_jit as jit;

struct Var<T> {
    r: jit::VarRef,
    _ty: PhantomData<T>,
}
