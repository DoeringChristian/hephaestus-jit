pub mod backend;
mod compiler;
mod data;
mod graph;
pub mod ir;
mod op;
pub mod trace;
mod vartype;

#[cfg(test)]
mod test;

pub use trace as tr;

// fn main() {
//     let device = backend::Device::vulkan(0).unwrap();
//
//     let i = trace::index(10);
//     let j = trace::index(5);
//
//     j.add(&trace::literal(1u32)).scatter(&i, &j);
//
//     j.schedule();
//
//     dbg!(&i);
//     dbg!(&j);
//
//     let graph = trace::compile();
//     dbg!(&graph);
//     graph.launch_slow(&device);
//
//     dbg!(graph.n_passes());
//
//     dbg!(&i.data().buffer().unwrap().to_host::<u32>());
//     dbg!(&j.data().buffer().unwrap().to_host::<u32>());
//     with_trace(|t| {
//         dbg!(&t);
//     });
// }
