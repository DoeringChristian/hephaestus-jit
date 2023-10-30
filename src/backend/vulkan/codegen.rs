use std::collections::HashMap;

use crate::trace::{Trace, VarId, VarType};
use rspirv::binary::{Assemble, Disassemble};
use rspirv::spirv;

use super::param_layout::ParamLayout;

// fn ty(ty: &VarType) ->

pub fn assemble_trace(trace: &Trace, entry_point: &str) -> Result<Vec<u32>, rspirv::dr::Error> {
    let param_layout = ParamLayout::generate(trace);
    dbg!(&param_layout);

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

    let void = b.type_void();
    let voidf = b.type_function(void, vec![]);

    let u32_ty = b.type_int(32, 0);
    let v3u32_ty = b.type_vector(u32_ty, 3);
    let ptr_v3u32_ty = b.type_pointer(None, spirv::StorageClass::Input, v3u32_ty);
    let global_invocation_id = b.variable(ptr_v3u32_ty, None, spirv::StorageClass::Input, None);

    b.decorate(
        global_invocation_id,
        spirv::Decoration::BuiltIn,
        [rspirv::dr::Operand::BuiltIn(
            spirv::BuiltIn::GlobalInvocationId,
        )],
    );

    let mut types = HashMap::new();
    assemble_types(&mut b, &mut types, trace, &param_layout);

    let func = b.begin_function(
        void,
        None,
        spirv::FunctionControl::DONT_INLINE | spirv::FunctionControl::CONST,
        voidf,
    )?;

    b.begin_block(None)?;

    let vars = trace.vars.iter().map(|_| b.id()).collect::<Vec<_>>();

    assemble_vars(
        &mut b,
        trace,
        &vars,
        &types,
        &param_layout,
        global_invocation_id,
    )?;

    b.ret()?;
    b.end_function()?;
    b.entry_point(spirv::ExecutionModel::GLCompute, func, entry_point, vec![]);
    b.execution_mode(func, spirv::ExecutionMode::LocalSize, [1, 1, 1]);

    let module = b.module();
    print!("{}", module.disassemble());
    Ok(module.assemble())
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

struct SpirvBuilder {
    b: rspirv::dr::Builder,
    types: HashMap<VarId, u32>,
    spirv_vars: Vec<u32>,
}

impl SpirvBuilder {
    // TODO: Spirv Builder
}

fn assemble_types(
    b: &mut rspirv::dr::Builder,
    types: &mut HashMap<VarId, u32>,
    trace: &Trace,
    param_layout: &ParamLayout,
) {
    for varid in trace.var_ids() {
        let var = trace.var(varid);
        match var.op {
            crate::trace::Op::LoadArray => {
                let ty = spirv_ty(b, &var.ty);
                let u32_ty = b.type_int(32, 0);
                let array_len = b.constant_u32(u32_ty, param_layout.len() as _);
                let rta_ty = b.type_runtime_array(ty);
                let struct_ty = b.type_struct([rta_ty]);
                let array_ty = b.type_array(struct_ty, array_len);

                b.decorate(struct_ty, spirv::Decoration::Block, []);

                let ptr_ty = b.type_pointer(None, spirv::StorageClass::StorageBuffer, array_ty);
                types.insert(varid, ptr_ty);
            }
            _ => {}
        }
    }
}
fn assemble_vars(
    b: &mut rspirv::dr::Builder,
    trace: &Trace,
    vars: &[u32],
    types: &HashMap<VarId, u32>,
    param_layout: &ParamLayout,
    global_invocation_id: u32,
) -> Result<(), rspirv::dr::Error> {
    for varid in trace.var_ids() {
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
            crate::trace::Op::Scatter { dst, src, idx } => {
                let ty = spirv_ty(b, &var.ty);
                let ptr_ty = b.type_pointer(None, spirv::StorageClass::StorageBuffer, ty);
                let int_ty = b.type_int(32, 0);
                let buffer = b.constant_u32(int_ty, param_layout.buffer_idx(dst) as _);
                let elem = b.constant_u32(int_ty, 0);
                let ptr = b.access_chain(ptr_ty, None, vars[dst.0], [buffer, elem, vars[idx.0]])?;
                b.store(ptr, vars[src.0], None, None)?;
            }
            crate::trace::Op::Gather { src, idx } => {
                let ty = spirv_ty(b, &var.ty);
                let ptr_ty = b.type_pointer(None, spirv::StorageClass::StorageBuffer, ty);
                let int_ty = b.type_int(32, 0);
                let buffer = b.constant_u32(int_ty, param_layout.buffer_idx(src) as _);
                let elem = b.constant_u32(int_ty, 0);
                let ptr = b.access_chain(ptr_ty, None, vars[src.0], [buffer, elem, vars[idx.0]])?;
                b.load(ty, Some(vars[varid.0]), ptr, None, None)?;
            }
            crate::trace::Op::Index => {
                let u32_ty = b.type_int(32, 0);
                let ptr_ty = b.type_pointer(None, spirv::StorageClass::Input, u32_ty);
                let u32_0 = b.constant_u32(u32_ty, 0);
                let ptr = b.access_chain(ptr_ty, None, global_invocation_id, [u32_0])?;
                b.load(u32_ty, Some(vars[varid.0]), ptr, None, None)?;
            }
            crate::trace::Op::Const { data } => todo!(),
            crate::trace::Op::LoadArray => {
                // let ty = spirv_ty(b, &var.ty);
                // let u32_ty = b.type_int(32, 0);
                // let array_len = b.constant_u32(u32_ty, param_layout.len() as _);
                // let rta_ty = b.type_runtime_array(ty);
                // let struct_ty = b.type_struct([rta_ty]);
                // let array_ty = b.type_array(struct_ty, array_len);
                //
                // b.decorate(struct_ty, spirv::Decoration::Block, []);
                //
                // let ptr_ty = b.type_pointer(None, spirv::StorageClass::StorageBuffer, array_ty);
                let ptr_ty = types[&varid];
                b.variable(
                    ptr_ty,
                    Some(vars[varid.0]),
                    spirv::StorageClass::StorageBuffer,
                    None,
                );
            }
        }
    }
    Ok(())
}
