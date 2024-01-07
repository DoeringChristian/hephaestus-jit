use std::array::IntoIter;

use crate::{backend, graph};

use crate::tr::{schedule_eval, with_trace, VarRef, SCHEDULE};

pub trait Traverse {
    // This operation flattens the structure to it's VarRef components
    // fn traverse<'a>(&'a self);
    fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef> {
        None.into_iter()
    }
}

impl Traverse for VarRef {
    fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef> {
        [self].into_iter()
    }
}
impl<const N: usize, T: Traverse> Traverse for [T; N] {
    fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef> {
        self.iter().flat_map(|i| i.traverse())
    }
}
impl<T: Traverse> Traverse for &[T] {
    fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef> {
        self.iter().flat_map(|i| i.traverse())
    }
}

macro_rules! impl_traverse_for_tuple {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param: Traverse),*> Traverse for ($($param,)*){
            fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef>{
                let ($($param,)*) = self;
                [].into_iter()
                $(.chain($param.traverse()))*
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

pub fn record<'a, Input, F>(f: F) -> impl FnMut(&backend::Device, Input) + 'a
where
    Input: Traverse + Clone,
    F: FnOnce(Input) + 'a,
{
    let mut graph = None;
    let mut f = Some(f);

    move |device, params: Input| {
        let param_vec = params.traverse().collect::<Vec<_>>();

        if graph.is_none() {
            // Swap out current schedule
            let tmp = SCHEDULE.with(|s| {
                let mut s = s.borrow_mut();
                std::mem::take(&mut *s)
            });

            let f = f.take().unwrap();
            f(params.clone());

            // Compile with params
            schedule_eval();
            graph = Some(SCHEDULE.with(|s| {
                let mut s = s.borrow_mut();
                let schedule = std::mem::take(&mut (*s));
                with_trace(|t| graph::compile(t, &schedule, &param_vec))
            }));

            // Swap in old schedule
            SCHEDULE.with(|s| {
                let mut s = s.borrow_mut();
                *s = tmp;
            });
        }
        graph.as_ref().unwrap().launch_with(device, &param_vec);
    }
}
