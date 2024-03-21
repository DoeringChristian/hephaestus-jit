use std::array::IntoIter;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::{backend, graph, tr};

use crate::tr::{schedule_eval, with_trace, VarRef, TS};

pub trait Traverse {
    // This operation flattens the structure to it's VarRef components
    // fn traverse<'a>(&'a self);
    fn traverse(&self, vec: &mut Vec<VarRef>);
}
pub trait Construct {
    fn construct(iter: &mut impl Iterator<Item = VarRef>) -> Self;
}

impl Traverse for VarRef {
    fn traverse(&self, vec: &mut Vec<VarRef>) {
        vec.push(self.clone())
    }
}
impl<T: Traverse> Traverse for &T {
    fn traverse(&self, vec: &mut Vec<VarRef>) {
        (**self).traverse(vec);
    }
}
// impl Traverse for &VarRef {
//     fn traverse(&self, vec: &mut Vec<VarRef>) {
//         vec.push((*self).clone())
//     }
// }
impl Construct for VarRef {
    fn construct(iter: &mut impl Iterator<Item = VarRef>) -> Self {
        iter.next().unwrap()
    }
}
impl<const N: usize, T: Traverse> Traverse for [T; N] {
    fn traverse(&self, vec: &mut Vec<VarRef>) {
        for i in self {
            i.traverse(vec);
        }
    }
}
impl<T: Traverse> Traverse for &[T] {
    fn traverse(&self, vec: &mut Vec<VarRef>) {
        for i in *self {
            i.traverse(vec);
        }
    }
}

macro_rules! impl_traverse_for_tuple {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param: Traverse),*> Traverse for ($($param,)*){
            fn traverse(&self, vec: &mut Vec<VarRef>) {
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

pub trait WrapInput<Input, Output> {
    // type Input;
    fn wrap_input(self) -> impl Fn(Input) -> Output + 'static;
}

macro_rules! impl_wrap_input {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param,)* Output, Fin> WrapInput<($($param,)*), Output> for Fin
        where
            Fin: Fn($($param,)*) -> Output + 'static,
        {
            // type Input = ($($param,)*);
            fn wrap_input(self) -> impl Fn(($($param,)*)) -> Output + 'static{
                move |input|{
                    let ($($param,)*) = input;
                    self($($param),*)
                }
            }
        }
    };
}

impl_wrap_input!();
impl_wrap_input!(A);
impl_wrap_input!(A, B);
impl_wrap_input!(A, B, C);
impl_wrap_input!(A, B, C, D);
impl_wrap_input!(A, B, C, D, E);
impl_wrap_input!(A, B, C, D, E, F);
impl_wrap_input!(A, B, C, D, E, F, G);
impl_wrap_input!(A, B, C, D, E, F, G, H);
impl_wrap_input!(A, B, C, D, E, F, G, H, I);
impl_wrap_input!(A, B, C, D, E, F, G, H, I, J);
impl_wrap_input!(A, B, C, D, E, F, G, H, I, J, K);
impl_wrap_input!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_wrap_input!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_wrap_input!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_wrap_input!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_wrap_input!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

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

pub fn record<Input, Output, F>(
    f: F,
) -> impl Fn(&backend::Device, Input) -> Option<(Output, backend::Report)>
where
    Input: Traverse,
    Output: Traverse + Construct,
    F: WrapInput<Input, Output>,
{
    static FCACHE: Lazy<Mutex<FCache>> = Lazy::new(|| Mutex::new(FCache::default()));

    let f = f.wrap_input();
    let f = Mutex::new(f);
    move |device: &backend::Device, input: Input| {
        let mut f = f.lock().unwrap();
        FCACHE.lock().unwrap().call(&mut *f, device, input)
    }
}

#[derive(Default)]
pub struct FCache {
    graphs: HashMap<u64, graph::Graph>,
}
impl FCache {
    pub fn call<Input, Output, F>(
        &mut self,
        f: &mut F,
        device: &backend::Device,
        input: Input,
    ) -> Option<(Output, backend::Report)>
    where
        Input: Traverse,
        Output: Traverse + Construct,
        F: Fn(Input) -> Output + 'static,
    {
        // Traverse Input
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

        // Calculate hash of function + input
        let mut hasher = DefaultHasher::new();

        // Calculate hash of function type
        std::any::TypeId::of::<F>().hash(&mut hasher);

        // Between function calls, the size or type of input variables might change.
        // To this end we keep a chache of graphs.
        // We calculate the hash of the inputs by using their resource descriptors, as they
        // uniquely identify the type and extent of the variable.
        with_trace(|trace| {
            for var in &inputs {
                trace.var(var.id()).resource_desc().hash(&mut hasher);
            }
        });
        let hash = hasher.finish();

        // If the correct graph could not be found, compile and insert it into the cache.
        if !self.graphs.contains_key(&hash) {
            let output = f(input);

            let mut outputs = vec![];
            output.traverse(&mut outputs);

            // Compile with params
            for v in &outputs {
                v.schedule();
            }
            schedule_eval();

            let graph = TS.with(|s| {
                let mut s = s.borrow_mut();
                let ts = std::mem::take(&mut (*s));
                graph::compile(&ts, &inputs, &outputs)
            });

            self.graphs.insert(hash, graph);
        }

        // Get the correct graph, launch it and construct the output struct.
        let graph = &self.graphs[&hash];
        let (report, output) = graph.launch_with(device, &inputs)?;
        let mut output = output.into_iter();
        Some((Output::construct(&mut output), report))
    }
}

#[cfg(test)]
mod test {
    use hephaestus_macros::{recorded, Construct, Traverse};
    use once_cell::sync::Lazy;

    use crate::backend::Device;
    use crate::record::FCache;
    use crate::{literal, VarRef};

    use super::WrapInput;

    #[test]
    fn derive_macros() {
        let fcache = FCache::default();

        let func = |x: VarRef, y: VarRef| {
            return x;
        };
        let func = func.wrap_input();

        #[derive(Traverse)]
        struct Test1<'b> {
            a: VarRef,
            b: VarRef,
            c: &'b [VarRef],
        }

        #[derive(Traverse)]
        struct Test2<'b>(VarRef, VarRef, &'b [VarRef]);

        #[derive(Traverse, Construct)]
        struct Test3 {
            a: VarRef,
            b: VarRef,
        }

        #[derive(Traverse, Construct)]
        struct Test4(VarRef, VarRef);
    }

    #[test]
    fn attribute_macros() {
        #[recorded]
        fn recording(x: VarRef, y: VarRef) -> VarRef {
            x.add(&literal(1))
        }
    }
}
