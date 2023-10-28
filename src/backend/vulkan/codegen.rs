use crate::trace::{Trace, VarId, VarType};
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

    for var in trace.var_ids() {
        assemble_var(&mut b, trace, var, &vars, &param_layout)?;
    }

    b.ret()?;
    b.end_function()?;

    print!("{}", b.module().disassemble());

    todo!()
}
fn spirv_ty(b: &mut rspirv::dr::Builder, ty: &VarType) -> u32 {
    match ty {
        VarType::Void => b.type_void(),
        VarType::Bool => b.type_bool(),
        VarType::I8 => b.type_int(8, 1),
        VarType::U8 => b.type_int(8, 0),
        VarType::I16 => b.type_int(16, 1),
        VarType::U16 => b.type_int(16, 0),
        VarType::I32 => b.type_int(32, 1),
        VarType::U32 => b.type_int(32, 0),
        VarType::I64 => b.type_int(64, 1),
        VarType::U64 => b.type_int(64, 0),
        VarType::F32 => b.type_float(32),
        VarType::F64 => b.type_float(64),
    }
}
fn isfloat(ty: &VarType) -> bool {
    match ty {
        VarType::F32 | VarType::F64 => true,
        _ => false,
    }
}
fn isint(ty: &VarType) -> bool {
    match ty {
        VarType::I8
        | VarType::U8
        | VarType::I16
        | VarType::U16
        | VarType::I32
        | VarType::U32
        | VarType::I64
        | VarType::U64 => true,
        _ => false,
    }
}
fn stb_ptr_ty(b: &mut rspirv::dr::Builder) {
    // b.type_pointer(spirv::StorageClass)
}
fn assemble_var(
    b: &mut rspirv::dr::Builder,
    trace: &Trace,
    varid: VarId,
    vars: &[u32],
    param_layout: &ParamLayout,
) -> Result<(), rspirv::dr::Error> {
    let var = trace.var(varid);
    match var.op {
        crate::trace::Op::Nop => {}
        crate::trace::Op::Add { lhs, rhs } => {
            if isint(&var.ty) {
                let ty = spirv_ty(b, &var.ty);
                b.i_add(ty, Some(vars[varid.0]), vars[lhs.0], vars[rhs.0])?;
            } else if isfloat(&var.ty) {
                let ty = spirv_ty(b, &var.ty);
                b.f_add(ty, Some(vars[varid.0]), vars[lhs.0], vars[rhs.0])?;
            } else {
                todo!()
            }
        }
        crate::trace::Op::Scatter { dst, src, idx } => todo!(),
        crate::trace::Op::Gather { src, idx } => todo!(),
        crate::trace::Op::Index => todo!(),
        crate::trace::Op::Const { data } => todo!(),
        crate::trace::Op::LoadArray => {
            // let ptr =  b.access_chain()
            todo!()
        }
    }
    Ok(())
}
