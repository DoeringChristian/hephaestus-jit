use std::sync::Arc;

use crate::scheduler::eval;

use self::backend::Device;
use self::scheduler::Scheduler;
use self::trace::with_trace;

pub mod backend;
mod data;
pub mod ir;
mod op;
mod scheduler;
mod trace;
mod vartype;

fn main() {
    with_trace(|t| t.device = Some(backend::Device::vulkan(0).unwrap()));

    let i = trace::index(10);
    let mut scheduler = Scheduler::default();
    with_trace(|t| {
        scheduler.collect(t, i.0);
    });
    with_trace(|t| eval(t, &[i.0]));
    dbg!(&scheduler);
}
