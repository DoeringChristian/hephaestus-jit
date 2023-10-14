use std::sync::Arc;

use crate::backend::Parameters;
use crate::trace::{Op, Trace, Var, VarType};

use self::backend::Device;

mod backend;
mod trace;
mod tracer;

fn main() {
    let device = Device::cuda(0).unwrap();
    let input_buffer = Arc::new(device.create_buffer(10).unwrap());
    let output_buffer = Arc::new(device.create_buffer(10).unwrap());

    let mut trace = Trace::default();

    let input = trace.push_var(Var { ty: VarType::Array });
    let output = trace.push_var(Var { ty: VarType::Array });

    let lhs = trace.push_var(Var { ty: VarType::U32 });
    let rhs = trace.push_var(Var { ty: VarType::U32 });
    let dst = trace.push_var(Var { ty: VarType::U32 });

    let idx = trace.push_var(Var { ty: VarType::U32 });

    trace.push_op(Op::Index { dst: idx });
    trace.push_op(Op::Gather {
        dst: lhs,
        src: input,
        idx,
    });
    trace.push_op(Op::Gather {
        dst: rhs,
        src: input,
        idx,
    });
    trace.push_op(Op::Add { lhs, rhs, dst });
    trace.push_op(Op::Scatter {
        dst: output,
        src: dst,
        idx,
    });

    device
        .execute_trace(
            &trace,
            Parameters {
                size: 1,
                buffers: vec![input_buffer, output_buffer.clone()],
            },
        )
        .unwrap();

    dbg!(output_buffer.to_host().unwrap());

    dbg!(input);
    dbg!(device);
}
