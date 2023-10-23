use std::sync::Arc;

use crate::backend::Parameters;
use crate::trace::{Op, Trace, Var, VarType};

use self::backend::Device;

pub mod backend;
pub mod trace;
mod tracer;

fn main() {
    let device = Device::cuda(0).unwrap();
    let output_buffer = device.create_array(10).unwrap();

    let mut trace = Trace::default();

    let output = trace.push_var(Var {
        // ty: VarType::Array,
        ..Default::default()
    });
    let c = trace.push_var(Var {
        ty: VarType::U32,
        ..Default::default()
    });
    let idx = trace.push_var(Var {
        ty: VarType::U32,
        ..Default::default()
    });

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
                arrays: vec![output_buffer.clone()],
            },
        )
        .unwrap();

    dbg!(output_buffer.to_host().unwrap());

    dbg!(device);
}
