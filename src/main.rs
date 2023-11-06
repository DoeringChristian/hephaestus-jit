use std::sync::Arc;

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
    trace::eval(&[&i]);
    dbg!(&i.data().buffer().unwrap().to_host::<u32>());
}
