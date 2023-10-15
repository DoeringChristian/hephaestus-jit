use std::sync::Arc;

use crate::backend::Parameters;
use crate::trace::{Op, Trace, Var, VarType};

use self::backend::Device;

#[macro_use]
extern crate pest_derive;

mod backend;
mod frontend;
mod trace;
mod tracer;

fn main() {
    let device = Device::cuda(0).unwrap();
    let input_buffer = Arc::new(device.create_buffer(10).unwrap());
    let output_buffer = Arc::new(device.create_buffer(10).unwrap());

    let mut trace = Trace::default();

    let output = trace.push_var(Var { ty: VarType::Array });
    let c = trace.push_var(Var { ty: VarType::U32 });
    let idx = trace.push_var(Var { ty: VarType::U32 });

    trace.push_op(Op::Index { dst: idx });
    trace.push_op(Op::Const { dst: c, data: 1 });

    trace.push_op(Op::Scatter {
        dst: output,
        src: c,
        idx,
    });

    device
        .execute_trace(
            &trace,
            Parameters {
                size: 10,
                buffers: vec![output_buffer.clone()],
            },
        )
        .unwrap();

    dbg!(output_buffer.to_host().unwrap());

    dbg!(device);
}
