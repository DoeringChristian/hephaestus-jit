use std::sync::Arc;

use crate::backend::Parameters;
use crate::trace::{Op, Trace, Var, VarType};
use crate::tracer::Kernel;

use self::backend::Device;

pub mod backend;
pub mod trace;
mod tracer;

fn main() {
    let device = Device::cuda(0).unwrap();
    let output = device.create_array(10, VarType::U32).unwrap();

    let k = Kernel::default();

    {
        let output = k.array(&output);
        let idx = k.index(10);

        idx.scatter(&output, &idx);
    }

    k.launch(&device).unwrap();

    dbg!(output.to_host().unwrap());
}
