use std::sync::Arc;

use self::backend::Device;
use self::scheduler::Scheduler;
use self::trace::with_trace;

pub mod backend;
pub mod ir;
mod op;
mod scheduler;
mod trace;
mod vartype;

fn main() {
    let i = trace::index(10);
    let mut scheduler = Scheduler::default();
    with_trace(|t| {
        scheduler.collect(t, i.0);
    });
    dbg!(&scheduler);
}
