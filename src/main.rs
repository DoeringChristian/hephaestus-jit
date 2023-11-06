use std::sync::Arc;

use self::backend::Device;

pub mod backend;
pub mod ir;
mod op;
mod trace;
mod vartype;

use trace as jit;

fn main() {
    let i = jit::index(10);
}
