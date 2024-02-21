use std::array::IntoIter;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::{backend, graph};

use crate::tr::{schedule_eval, with_trace, VarRef, TS};

pub trait Traverse {
    // This operation flattens the structure to it's VarRef components
    // fn traverse<'a>(&'a self);
    fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef> {
        None.into_iter()
    }
}
pub trait Construct {
    fn construct(iter: &mut impl Iterator<Item = VarRef>) -> Self;
}

impl Traverse for VarRef {
    fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef> {
        [self].into_iter()
    }
}
impl Construct for VarRef {
    fn construct(iter: &mut impl Iterator<Item = VarRef>) -> Self {
        iter.next().unwrap()
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

macro_rules! impl_construct_for_tuple {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param: Construct),*> Construct for ($($param,)*){
            fn construct(iter: &mut impl Iterator<Item = VarRef>) -> Self{
                ($($param::construct(iter),)*)
            }
            // fn traverse<'a>(&'a self) -> impl Iterator<Item = &'a VarRef>{
            //     let ($($param,)*) = self;
            //     [].into_iter()
            //     $(.chain($param.traverse()))*
            // }
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

pub fn record<'a, Input, Output, F>(f: F) -> impl FnMut(&backend::Device, Input) -> Output + 'a
where
    Input: Traverse + Clone,
    Output: Traverse + Construct + Clone,
    F: FnMut(Input) -> Output + 'a,
{
    // TODO: make this a cache
    let mut graphs = HashMap::new();
    let f = RefCell::new(f);
    // let mut f = Some(f);

    move |device, input: Input| {
        let input_vec = input.traverse().collect::<Vec<_>>();

        let mut hasher = DefaultHasher::new();
        with_trace(|trace| {
            for var in &input_vec {
                trace.var(var.id()).resource_desc().hash(&mut hasher);
            }
        });
        let hash = hasher.finish();

        if !graphs.contains_key(&hash) {
            // Swap out current schedule
            let tmp = TS.with(|s| {
                let mut s = s.borrow_mut();
                std::mem::take(&mut *s)
            });

            let mut f = f.borrow_mut();
            let output = f(input.clone());

            let output_vec = output.traverse().collect::<Vec<_>>();

            for v in &output_vec {
                v.schedule();
            }

            // Compile with params
            schedule_eval();
            graphs.insert(
                hash,
                TS.with(|s| {
                    let mut s = s.borrow_mut();
                    let schedule = std::mem::take(&mut (*s));
                    with_trace(|t| graph::compile(t, &schedule, &input_vec, &output_vec))
                }),
            );

            // Swap in old schedule
            TS.with(|s| {
                let mut s = s.borrow_mut();
                *s = tmp;
            });
        }
        let graph = &graphs[&hash];
        let (report, output) = graph.launch_with(device, &input_vec);
        let mut output = output.into_iter();
        Output::construct(&mut output)
    }
}

#[macro_export]
macro_rules! loop_record {
    ([$($vars:ident),*] while $cond:ident $content:block) => {
        let cond_vars = [&$cond, $(&$vars),*];
        let (loop_start, state) = $crate::tr::loop_start(cond_vars.as_slice());

        let mut itr = state.into_iter();
        $cond = itr.next().unwrap();
        $($vars = itr.next().unwrap())*;

        $content

        let cond_vars = [&$cond, $(&$vars),*];
        let state = $crate::tr::loop_end(&loop_start, cond_vars.as_slice());

        let mut itr = state.into_iter();
        $cond = itr.next().unwrap();
        $($vars = itr.next().unwrap())*;
    };
}
