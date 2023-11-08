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
    let j = trace::index(5);

    j.add(&trace::literal(1u32)).scatter(&i, &j);

    dbg!(&i);
    dbg!(&j);

    let graph = trace::compile();
    dbg!(&graph);
    graph.launch_slow(&device);

    dbg!(graph.n_passes());

    dbg!(&i.data().buffer().unwrap().to_host::<u32>());
    dbg!(&j.data().buffer().unwrap().to_host::<u32>());
}
