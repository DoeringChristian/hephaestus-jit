use std::sync::Arc;

use crate::trace::VarType;

use self::backend::Device;

pub mod backend;
pub mod trace;
mod tracer;

use tracer as jit;

fn main() {
    let device = Device::vulkan(0).unwrap();
    let output = device.create_array(10, VarType::U32).unwrap();

    {
        let output = jit::array(&output);
        let idx = jit::index(10);

        idx.scatter(&output, &idx);
    }

    jit::launch(&device).unwrap();

    dbg!(output.to_host::<u8>().unwrap());
}
