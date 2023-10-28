use crate::trace::{OpId, Trace, VarId, VarType};
use rspirv::binary::Disassemble;
use rspirv::spirv;

use super::param_layout::ParamLayout;

// fn ty(ty: &VarType) ->

pub fn assemble_trace(trace: &Trace, entry_point: &str) -> Result<(), rspirv::dr::Error> {
    let param_layout = ParamLayout::generate(trace);

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::Simple);

    let void = b.type_void();
    let voidf = b.type_function(void, vec![void]);

    b.begin_function(
        void,
        None,
        spirv::FunctionControl::DONT_INLINE | spirv::FunctionControl::CONST,
        voidf,
    )?;

    b.begin_block(None)?;

    let vars = trace.vars.iter().map(|_| b.id()).collect::<Vec<_>>();

    for opid in trace.op_ids() {
        assemble_op(&mut b, trace, opid, &vars, &param_layout)?;
    }

    b.ret()?;
    b.end_function()?;

    print!("{}", b.module().disassemble());

    todo!()
}
fn assemble_op(
    b: &mut rspirv::dr::Builder,
    trace: &Trace,
    opid: OpId,
    vars: &[u32],
    param_layout: &ParamLayout,
) -> Result<(), rspirv::dr::Error> {
    let op = trace.op(opid);
    match op {
        crate::trace::Op::Add { dst, lhs, rhs } => match trace.var_ty(*dst) {
            VarType::I8 => {
                let ty = b.type_int(32, 1);
                b.i_add(ty, Some(vars[dst.0]), vars[lhs.0], vars[rhs.0])?;
            }
            VarType::U8 => todo!(),
            VarType::I16 => todo!(),
            VarType::U16 => todo!(),
            VarType::I32 => todo!(),
            VarType::U32 => todo!(),
            VarType::I64 => todo!(),
            VarType::U64 => todo!(),
            VarType::F32 => todo!(),
            VarType::F64 => todo!(),
            _ => todo!(),
        },
        crate::trace::Op::Scatter { dst, src, idx } => todo!(),
        crate::trace::Op::Gather { dst, src, idx } => todo!(),
        crate::trace::Op::Index { dst } => todo!(),
        crate::trace::Op::Const { dst, data } => todo!(),
    };
    Ok(())
}
