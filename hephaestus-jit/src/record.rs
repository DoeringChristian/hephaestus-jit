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
impl Traverse for &VarRef {
    fn traverse(&self, vec: &mut Vec<VarRef>) {
        vec.push((*self).clone())
    }
}
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

pub struct Func<Input, Output> {
    graphs: Mutex<HashMap<u64, graph::Graph>>,
    f: Mutex<Box<dyn FnMut(Input) -> Output>>,
    _in: PhantomData<Input>,
    _out: PhantomData<Output>,
}

impl<Input, Output> Func<Input, Output>
where
    Input: Traverse + Clone,
    Output: Traverse + Construct + Clone + 'static,
{
    pub fn new(f: impl FnMut(Input) -> Output + 'static) -> Self {
        Self {
            graphs: Mutex::new(HashMap::new()),
            f: Mutex::new(Box::new(f)),
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
        let (report, output) = graph.launch_with(device, inputs.as_slice())?;
        let mut output = output.into_iter();
        Some((report, Output::construct(&mut output)))
    }
}

pub fn record<Input, Output, F>(f: F) -> impl Fn(&backend::Device, Input) -> Output
where
    Input: Traverse + Clone,
    Output: Traverse + Construct + Clone,
    F: Recordable<Input, Output>,
{
    f.record()
}

pub trait WrapInput<Input, Output> {
    type Input;
    fn wrap_input(self) -> impl Fn(Input) -> Output;
}

macro_rules! impl_wrap_input {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param,)* Output, Fin> WrapInput<($($param,)*), Output> for Fin
        where
            $($param: Traverse + Clone,)*
            Output: Traverse + Construct + Clone,
            Fin: Fn($($param,)*) -> Output,
        {
            type Input = ($($param,)*);
            fn wrap_input(self) -> impl Fn(Self::Input) -> Output{
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

pub trait Recordable<Input, Output> {
    type Input;
    fn record(self) -> impl Fn(&backend::Device, Input) -> Output;
    fn func(self) -> Func<Input, Output>;
}

macro_rules! impl_recordable {
    ($($param:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($param,)* Output, Fin> Recordable<($($param,)*), Output> for Fin
        where
            $($param: Traverse + Clone,)*
            Output: Traverse + Construct + Clone + 'static,
            Fin: Fn($($param,)*) -> Output + 'static,
        {
            type Input = ($($param,)*);
            fn record(self) -> impl Fn(&backend::Device, Self::Input) -> Output {
                let func = self.func().func();

                move |device: &backend::Device, input: Self::Input| func(device, input)
            }
            fn func(self) -> Func<Self::Input, Output>{
                Func::new(move |input: Self::Input| {
                    let ($($param,)*) = input;
                    self($($param),*)
                })
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

pub struct FuncCache {
    graphs: HashMap<u64, graph::Graph>,
}
impl FuncCache {
    pub fn call<Input, Output, F>(
        &mut self,
        f: F,
        device: &backend::Device,
        input: Input,
    ) -> Option<(backend::Report, Output)>
    where
        Input: Traverse,
        Output: Traverse + Construct,
        F: FnOnce(Input) -> Output + 'static,
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

        let graph = &self.graphs[&hash];
        let (report, output) = graph.launch_with(device, &inputs)?;
        let mut output = output.into_iter();
        Some((report, Output::construct(&mut output)))
    }
}

#[cfg(test)]
mod test {
    use hephaestus_macros::{recorded, Construct, Traverse};
    use once_cell::sync::Lazy;

    use crate::backend::Device;
    use crate::record::{FuncCache, Recordable};
    use crate::{literal, VarRef};

    use super::Func;
    use super::WrapInput;

    #[test]
    fn derive_macros() {
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
        // #[record]
        // let _recording =
        // Func::<(VarRef, VarRef), VarRef, fn(VarRef, VarRef) -> VarRef>::new(recording);
        // let _recording = Func::new(recording);
        fn recorded(device: &Device, x: VarRef, y: VarRef) -> VarRef {
            fn recording(x: VarRef, y: VarRef) -> VarRef {
                x.add(&literal(1))
            }
            const RECORDING: Lazy<Func<(VarRef, VarRef), VarRef>> = Lazy::new(|| recording.func());
            RECORDING.call(device, (x, y))
        }

        #[recorded]
        fn recording(x: VarRef, y: VarRef) -> VarRef {
            x.add(&literal(1))
        }
    }
}
