use std::sync::Arc;

use self::backend::Device;
use self::compiler::Compiler;
use self::trace::with_trace;

pub mod backend;
mod compiler;
mod data;
mod graph;
pub mod ir;
mod op;
mod trace;
mod vartype;

fn main() {
    let device = backend::Device::vulkan(0).unwrap();
    with_trace(|t| t.device = Some(device.clone()));

    let i = trace::index(10);
    let j = trace::index(20);

    i.schedule();

    let graph = trace::compile();
    graph.launch_slow(&device);

    // trace::eval(&[&i]);
    dbg!(&i.data().buffer().unwrap().to_host::<u32>());
}
