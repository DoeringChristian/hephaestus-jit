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

    let i = trace::index(10);
    let idx = trace::index(5);

    let j = i.gather(&idx);

    i.schedule();
    j.schedule();

    let graph = trace::compile();
    graph.launch_slow(&device);

    dbg!(graph.n_passes());

    dbg!(&i.data().buffer().unwrap().to_host::<u32>());
    dbg!(&j.data().buffer().unwrap().to_host::<u32>());
}
