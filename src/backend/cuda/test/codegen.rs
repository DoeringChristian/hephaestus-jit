use crate::backend::cuda::codegen::assemble_trace;
use crate::backend::cuda::param_layout::ParamLayout;
use crate::trace::*;

#[test]
fn gather() {
    let mut trace = Trace::default();

    let array = trace.push_var(Var {
        ty: VarType::Array,
        external: Some(0),
    });

    let x = trace.push_var(Var {
        ty: VarType::U8,
        ..Default::default()
    });
    let idx = trace.push_var(Var {
        ty: VarType::U32,
        ..Default::default()
    });

    let gather = trace.push_op(Op::Gather {
        src: array,
        dst: x,
        idx,
    });

    let layout = ParamLayout {
        size: 0,
        arrays: 1,
        n_params: 2,
    };

    let mut asm = String::new();
    assemble_trace(&mut asm, &trace, gather, "global", layout).unwrap();
    insta::assert_snapshot!(asm);
}

#[test]
fn scatter() {
    let mut trace = Trace::default();

    let array = trace.push_var(Var {
        ty: VarType::Array,
        external: Some(0),
    });

    let x = trace.push_var(Var {
        ty: VarType::U8,
        ..Default::default()
    });
    let idx = trace.push_var(Var {
        ty: VarType::U32,
        ..Default::default()
    });

    let scatter = trace.push_op(Op::Scatter {
        src: x,
        dst: array,
        idx,
    });

    let layout = ParamLayout {
        size: 0,
        arrays: 1,
        n_params: 2,
    };

    let mut asm = String::new();
    assemble_trace(&mut asm, &trace, scatter, "global", layout).unwrap();
    insta::assert_snapshot!(asm);
}
