use std::any::TypeId;
use std::collections::HashMap;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::tr;
use crate::VarRef;

///
/// Layout is similar to VarType, but for external types i.e. struct od arrays
///
#[derive(Debug, Hash, PartialEq, Eq)]
pub enum Layout {
    Elem,
    Vec { ty: &'static Layout, num: usize },
    Tuple { tys: Vec<&'static Layout> },
}

impl Layout {
    pub fn elem() -> &'static Self {
        let mut hasher = DefaultHasher::new();
        // Function uid
        0u32.hash(&mut hasher);
        let hash = hasher.finish();

        LAYOUT_CACHE
            .lock()
            .unwrap()
            .entry(hash)
            .or_insert_with(|| Box::leak(Box::new(Self::Elem)))
    }
    pub fn tuple(tys: &[&'static Layout]) -> &'static Layout {
        let mut hasher = DefaultHasher::new();
        // Function id
        1u32.hash(&mut hasher);
        tys.hash(&mut hasher);
        let hash = hasher.finish();

        LAYOUT_CACHE
            .lock()
            .unwrap()
            .entry(hash)
            .or_insert_with(|| Box::leak(Box::new(Self::Tuple { tys: tys.to_vec() })))
    }
}

static LAYOUT_CACHE: Lazy<Mutex<HashMap<u64, &'static Layout>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub trait Traverse {
    /// Flattens the collection of variables, and records the layout.
    ///
    /// The layout vector records the number of direct elemenmts in a collection.
    /// Therefore 0 for VarRef and N for [T; N] where T: Traverse.
    /// This is used by the [`Construct`] trait to allocate vectors.
    ///
    fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>);

    ///
    /// Creates a compact composite variable from the variables in this container.
    /// This corresponds to transposing the type from a struct of arrays to an array of structs.
    ///
    fn ravel(&self) -> VarRef;
}
pub trait Construct: Sized {
    fn construct(
        vars: &mut impl Iterator<Item = VarRef>,
        layout: &mut impl Iterator<Item = usize>,
    ) -> Self;

    fn unravel(var: impl AsRef<VarRef>) -> Self;
}

impl Traverse for VarRef {
    fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>) {
        layout.push(0);
        vars.push(self.clone())
    }
    fn ravel(&self) -> VarRef {
        self.clone()
    }
}
impl<T: Traverse> Traverse for &T {
    fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>) {
        layout.push(0);
        (**self).traverse(vars, layout);
    }

    fn ravel(&self) -> VarRef {
        (*self).ravel()
    }
}

impl Construct for VarRef {
    fn construct(
        vars: &mut impl Iterator<Item = VarRef>,
        layout: &mut impl Iterator<Item = usize>,
    ) -> Self {
        assert_eq!(layout.next().unwrap(), 0);
        vars.next().unwrap()
    }
    fn unravel(var: impl AsRef<VarRef>) -> Self {
        var.as_ref().clone()
    }
}

impl<T: Traverse> Traverse for Vec<T> {
    fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>) {
        layout.push(self.len());
        for i in self {
            i.traverse(vars, layout);
        }
    }

    fn ravel(&self) -> VarRef {
        let refs = self.iter().map(|t| t.ravel()).collect::<Vec<_>>();
        tr::arr(&refs)
    }
}

impl<T: Construct> Construct for Vec<T> {
    fn construct(
        vars: &mut impl Iterator<Item = VarRef>,
        layout: &mut impl Iterator<Item = usize>,
    ) -> Self {
        let len = layout.next().unwrap();

        (0..len).map(|_| T::construct(vars, layout)).collect()
    }

    fn unravel(var: impl AsRef<VarRef>) -> Self {
        let var = var.as_ref();
        var.extract_all().map(|r| T::unravel(r)).collect::<Vec<_>>()
    }
}

impl<const N: usize, T: Traverse> Traverse for [T; N] {
    fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>) {
        layout.push(N);
        for i in self {
            i.traverse(vars, layout);
        }
    }
    fn ravel(&self) -> VarRef {
        let refs = self.iter().map(|t| t.ravel()).collect::<Vec<_>>();
        tr::arr(&refs)
    }
}
impl<T: Traverse> Traverse for &[T] {
    fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>) {
        layout.push(self.len());
        for i in *self {
            i.traverse(vars, layout);
        }
    }

    fn ravel(&self) -> VarRef {
        let refs = self.iter().map(|t| t.ravel()).collect::<Vec<_>>();
        tr::arr(&refs)
    }
}

macro_rules! impl_traverse_for_tuple {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param: Traverse),*> Traverse for ($($param,)*){
            fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>) {
                let i = layout.len();
                layout.push(0);

                let mut len = 0;
                let ($($param,)*) = self;
                $(
                    len += 1;
                    $param.traverse(vars, layout);
                )*
                layout[i] = len;
            }
            fn ravel(&self) -> VarRef {
                let ($($param,)*) = self;
                let refs = [$($param.ravel()),*];
                tr::composite(&refs)
            }
        }
    };
}
impl_traverse_for_tuple!();
impl_traverse_for_tuple!(A);
impl_traverse_for_tuple!(A, B);
impl_traverse_for_tuple!(A, B, C);
impl_traverse_for_tuple!(A, B, C, D);
impl_traverse_for_tuple!(A, B, C, D, E);
impl_traverse_for_tuple!(A, B, C, D, E, F);
impl_traverse_for_tuple!(A, B, C, D, E, F, G);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_traverse_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

macro_rules! impl_construct_for_tuple {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param: Construct),*> Construct for ($($param,)*){
            fn construct(vars: &mut impl Iterator<Item = VarRef>, layout: &mut impl Iterator<Item = usize>) -> Self{
            layout.next().unwrap();
                ($($param::construct(vars, layout),)*)
            }
            fn unravel(var: impl AsRef<VarRef>) -> Self {
                let var = var.as_ref();
                let mut iter = var.extract_all();
                ($($param::unravel(iter.next().unwrap()),)*)
            }
        }
    };
}
impl_construct_for_tuple!();
impl_construct_for_tuple!(A);
impl_construct_for_tuple!(A, B);
impl_construct_for_tuple!(A, B, C);
impl_construct_for_tuple!(A, B, C, D);
impl_construct_for_tuple!(A, B, C, D, E);
impl_construct_for_tuple!(A, B, C, D, E, F);
impl_construct_for_tuple!(A, B, C, D, E, F, G);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_construct_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
