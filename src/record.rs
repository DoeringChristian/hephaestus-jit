use std::array::IntoIter;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Mutex;

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

pub struct Func<Input, Output, F> {
    graphs: Mutex<HashMap<u64, graph::Graph>>,
    f: Mutex<F>,
    _in: PhantomData<Input>,
    _out: PhantomData<Output>,
}

impl<Input, Output, F> Func<Input, Output, F>
where
    Input: Traverse + Clone + 'static,
    Output: Traverse + Construct + Clone + 'static,
    F: FnMut(Input) -> Output,
{
    pub fn new(f: F) -> Self {
        Self {
            graphs: Mutex::new(HashMap::new()),
            f: Mutex::new(f),
            _in: PhantomData,
            _out: PhantomData,
        }
    }
    pub fn func(self) -> impl Fn(&backend::Device, Input) -> Output {
        move |device: &backend::Device, input: Input| {
            let _ = ();
            self.call(device, input)
        }
    }
    pub fn call(&self, device: &backend::Device, input: Input) -> Output {
        self.call_report(device, input).1
    }
    pub fn call_report(&self, device: &backend::Device, input: Input) -> (backend::Report, Output) {
        let input_vec = input.traverse().collect::<Vec<_>>();

        let mut hasher = DefaultHasher::new();
        with_trace(|trace| {
            for var in &input_vec {
                trace.var(var.id()).resource_desc().hash(&mut hasher);
            }
        });
        let hash = hasher.finish();

        if !self.graphs.lock().unwrap().contains_key(&hash) {
            // Swap out current schedule
            // TODO: think about this
            let tmp = TS.with(|s| {
                let mut s = s.borrow_mut();
                std::mem::take(&mut *s)
            });

            let mut f = self.f.lock().unwrap();
            let output = f(input.clone());

            let output_vec = output.traverse().collect::<Vec<_>>();

            for v in &output_vec {
                v.schedule();
            }

            // Compile with params
            schedule_eval();
            self.graphs.lock().unwrap().insert(
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
        let graph = &self.graphs.lock().unwrap()[&hash];
        let (report, output) = graph.launch_with(device, &input_vec);
        let mut output = output.into_iter();
        (report, Output::construct(&mut output))
    }
}

pub fn record<Input, Output, F>(f: F) -> impl Fn(&backend::Device, Input) -> Output
where
    Input: Traverse + Clone + 'static,
    Output: Traverse + Construct + Clone + 'static,
    F: Recordable<Input, Output>,
{
    f.record()
}

pub trait Recordable<Input, Output> {
    fn record(self) -> impl Fn(&backend::Device, Input) -> Output;
}

macro_rules! impl_recordable {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param,)* Output, Fin> Recordable<($($param,)*), Output> for Fin
        where
            $($param: Traverse + Clone + 'static,)*
            Output: Traverse + Construct + Clone + 'static,
            Fin: FnMut($($param,)*) -> Output,
        {
            fn record(mut self) -> impl Fn(&backend::Device, ($($param,)*)) -> Output {
                let func = Func::new(move |input: ($($param,)*)| {
                    let ($($param,)*) = input;
                    self($($param),*)
                }).func();

                move |device: &backend::Device, input: ($($param,)*)| func(device, input)
            }
        }
    };
}
impl_recordable!();
impl_recordable!(A);
impl_recordable!(A, B);
impl_recordable!(A, B, C);
impl_recordable!(A, B, C, D);
impl_recordable!(A, B, C, D, E);
impl_recordable!(A, B, C, D, E, F);
impl_recordable!(A, B, C, D, E, F, G);
impl_recordable!(A, B, C, D, E, F, G, H);
impl_recordable!(A, B, C, D, E, F, G, H, I);
impl_recordable!(A, B, C, D, E, F, G, H, I, J);
impl_recordable!(A, B, C, D, E, F, G, H, I, J, K);
impl_recordable!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_recordable!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_recordable!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_recordable!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_recordable!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

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
