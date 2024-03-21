#![feature(error_generic_member_access)]

pub mod backend;
mod compiler;
mod extent;
mod graph;
pub mod ir;
mod op;
mod prehashed;

#[macro_use]
pub mod record;
mod resource;
pub mod trace;
mod utils;
pub mod vartype;

#[cfg(test)]
mod test;

pub use trace as tr;

pub use backend::{vulkan, Device};
pub use graph::Graph;
pub use record::{Construct, Traverse};
pub use trace::*;
pub use vartype::AsVarType;

pub use hephaestus_macros::{recorded, AsVarType, Construct, Traverse};

pub use once_cell;
