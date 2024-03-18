use std::array::IntoIter;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Mutex;

use crate::{backend, graph, tr};

use crate::tr::{schedule_eval, with_trace, VarRef, TS};

pub trait Traverse {
    // This operation flattens the structure to it's VarRef components
    // fn traverse<'a>(&'a self);
    fn traverse<'a>(&'a self, vec: &mut Vec<&'a VarRef>) {}
}
pub trait Construct {
    fn construct(iter: &mut impl Iterator<Item = VarRef>) -> Self;
}

impl Traverse for VarRef {
    fn traverse<'a>(&'a self, vec: &mut Vec<&'a VarRef>) {
        vec.push(self)
    }
}
impl Traverse for &VarRef {
    fn traverse<'a>(&'a self, vec: &mut Vec<&'a VarRef>) {
        vec.push(*self)
    }
}
impl Construct for VarRef {
    fn construct(iter: &mut impl Iterator<Item = VarRef>) -> Self {
        iter.next().unwrap()
    }
}
impl<const N: usize, T: Traverse> Traverse for [T; N] {
    fn traverse<'a>(&'a self, vec: &mut Vec<&'a VarRef>) {
        for i in self {
            i.traverse(vec);
        }
    }
}
impl<T: Traverse> Traverse for &[T] {
    fn traverse<'a>(&'a self, vec: &mut Vec<&'a VarRef>) {
        for i in *self {
            i.traverse(vec);
        }
    }
}

macro_rules! impl_traverse_for_tuple {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param: Traverse),*> Traverse for ($($param,)*){
            fn traverse<'a>(&'a self, vec: &mut Vec<&'a VarRef>) {
                let ($($param,)*) = self;
                $($param.traverse(vec);)*
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
        self.call_report(device, input).unwrap().1
    }
    pub fn call_report(
        &self,
        device: &backend::Device,
        input: Input,
    ) -> Option<(backend::Report, Output)> {
        let mut inputs = vec![];
        input.traverse(&mut inputs);

        // We evaluate the inputs to the function, to not collect dependencies of input variables.
        // This might not be the best solution, but it solves some of the problems.
        for input in inputs.iter() {
            input.schedule();
        }
        // Evaluate all iput variables
        let graph = tr::compile();
        graph.launch(&device);

        // Between function calls, the size or type of input variables might change.
        // To this end we keep a chache of graphs.
        // We calculate the hash of the inputs by using their resource descriptors, as they
        // uniquely identify the type and extent of the variable.
        let mut hasher = DefaultHasher::new();
        with_trace(|trace| {
            for var in &inputs {
                trace.var(var.id()).resource_desc().hash(&mut hasher);
            }
        });
        let hash = hasher.finish();

        if !self.graphs.lock().unwrap().contains_key(&hash) {
            let mut f = self.f.lock().unwrap();
            let output = f(input.clone());

            let mut outputs = vec![];
            output.traverse(&mut outputs);

            for v in &outputs {
                v.schedule();
            }

            // Compile with params
            schedule_eval();
            self.graphs.lock().unwrap().insert(
                hash,
                TS.with(|s| {
                    let mut s = s.borrow_mut();
                    let ts = std::mem::take(&mut (*s));
                    graph::compile(&ts, &inputs, &outputs)
                }),
            );
        }
        let graph = &self.graphs.lock().unwrap()[&hash];
        let (report, output) = graph.launch_with(device, &inputs)?;
        let mut output = output.into_iter();
        Some((report, Output::construct(&mut output)))
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
            Fin: Fn($($param,)*) -> Output,
        {
            fn record(self) -> impl Fn(&backend::Device, ($($param,)*)) -> Output {
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
        let (loop_start, mut state) = $crate::tr::loop_start(cond_vars.as_slice());

        $cond = state.next().unwrap();
        $($vars = state.next().unwrap())*;

        $content

        let cond_vars = [&$cond, $(&$vars),*];
        let mut state = $crate::tr::loop_end(&loop_start, cond_vars.as_slice());

        $cond = state.next().unwrap();
        $($vars = state.next().unwrap())*;
    };
}
#[macro_export]
macro_rules! if_record {
    ([$($vars:ident),*] if $cond:ident $content:block) => {
        let cond_vars = [&$cond, $(&$vars),*];
        let (if_start, mut state) = $crate::tr::if_start(cond_vars.as_slice());

        $cond = state.next().unwrap();
        $($vars = state.next().unwrap())*;

        $content

        let cond_vars = [&$cond, $(&$vars),*];
        let mut state = $crate::tr::if_end(&if_start, cond_vars.as_slice());

        $cond = state.next().unwrap();
        $($vars = state.next().unwrap())*;
    };
}

#[cfg(test)]
mod test {
    use hephaestus_macros::Traverse;

    use crate::VarRef;

    #[test]
    fn traverse_macro() {
        #[derive(Traverse)]
        struct Test1<'b> {
            a: VarRef,
            b: VarRef,
            c: &'b [VarRef],
        }
        #[derive(Traverse)]
        struct Test2<'b>(VarRef, VarRef, &'b [VarRef]);
    }
}