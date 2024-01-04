use std::array::IntoIter;

use crate::{backend, graph};

use crate::tr::{schedule_eval, with_trace, VarRef, SCHEDULE};

struct Test {
    x: VarRef,
    y: VarRef,
}

pub trait Param {
    type Iterator: Iterator<Item = VarRef>;
    fn iter(&self) -> Self::Iterator;
}

impl Param for VarRef {
    type Iterator = core::array::IntoIter<VarRef, 1>;

    fn iter(&self) -> Self::Iterator {
        [self.clone()].into_iter()
    }
}

impl<A: Param, B: Param> Param for (A, B) {
    type Iterator = core::iter::Chain<A::Iterator, B::Iterator>;

    fn iter(&self) -> Self::Iterator {
        self.0.iter().chain(self.1.iter())
    }
}

pub trait Closure<F> {
    type In: Param;
    fn run(&mut self, input: Self::In);
}

impl<Input: Param, F: Send + Sync + 'static> Closure<fn(Input)> for F
where
    F: FnMut(Input),
{
    type In = Input;

    fn run(&mut self, input: Self::In) {
        self(input)
    }
}

pub fn record<Input: Param + Clone, F: Send + Sync + 'static + FnMut(Input)>(
    mut f: impl Closure<fn(Input), In = Input>,
) -> impl FnMut(&backend::Device, Input) {
    let mut graph = None;
    let mut ret = None;
    move |device, params| {
        if graph.is_none() {
            // Swap out current schedule
            let tmp = SCHEDULE.with(|s| {
                let mut s = s.borrow_mut();
                std::mem::take(&mut *s)
            });

            let param_vec = params.iter().collect::<Vec<_>>();

            ret = Some(f.run(params.clone()));

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
        dbg!(&graph);
        let param_vec = params.iter().collect::<Vec<_>>();
        graph.as_ref().unwrap().launch_with(device, &param_vec);
        ret.clone().unwrap()
    }
}
