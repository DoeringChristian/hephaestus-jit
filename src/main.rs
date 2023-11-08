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
    with_trace(|t| t.device = Some(backend::Device::vulkan(0).unwrap()));

    let i = trace::index(10);

    // trace::eval(&[&i]);
    // dbg!(&i.data().buffer().unwrap().to_host::<u32>());
}
