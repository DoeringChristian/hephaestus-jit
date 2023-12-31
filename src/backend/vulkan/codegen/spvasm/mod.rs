use super::CompileInfo;
use crate::ir::{VarId, IR};
use crate::vartype::{AsVarType, VarType};
use std::collections::HashSet;
use std::fmt::Write;

pub fn assemble_ir(ir: &IR, info: &CompileInfo, entry_point: &str) -> Option<Vec<u32>> {
    let mut asm = String::new();
    assemble_entry_point(&mut asm, ir, info, entry_point).unwrap();

    println!("{asm}");

    use spirv_tools::assembler;
    use spirv_tools::assembler::Assembler;
    let assembler = assembler::compiled::CompiledAssembler::default();

    let binary = assembler
        .assemble(
            &asm,
            assembler::AssemblerOptions {
                preserve_numeric_ids: false,
            },
        )
        .unwrap();

    Some(binary.as_words().to_vec())
}

#[derive(Clone, Copy, Debug)]
pub struct Reg(VarId);
impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%r{id}", id = self.0 .0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpvType(&'static VarType);
impl std::fmt::Display for SpvType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            VarType::Void => write!(f, "void"),
            VarType::Bool => write!(f, "bool"),
            VarType::I8 => write!(f, "i8"),
            VarType::U8 => write!(f, "u8"),
            VarType::I16 => write!(f, "i16"),
            VarType::U16 => write!(f, "u16"),
            VarType::I32 => write!(f, "i32"),
            VarType::U32 => write!(f, "u32"),
            VarType::I64 => write!(f, "i64"),
            VarType::U64 => write!(f, "u64"),
            VarType::F16 => write!(f, "f16"),
            VarType::F32 => write!(f, "f32"),
            VarType::F64 => write!(f, "f64"),
            VarType::Vec { ty, num } => {
                write!(f, "{ty}x{num}", ty = SpvType(ty))
            }
            VarType::Array { ty, num } => write!(f, "{ty}a{num}", ty = SpvType(ty)),
            VarType::Mat { ty, rows, cols } => write!(f, "{ty}m{rows}x{cols}", ty = SpvType(ty)),
            VarType::Struct { tys } => {
                write!(f, "struct_")?;
                for ty in tys {
                    write!(f, "{ty}", ty = SpvType(ty))?;
                }
                write!(f, "_")?;
                Ok(())
            }
        }
    }
}

pub fn assemble_entry_point(
    s: &mut String,
    ir: &IR,
    info: &CompileInfo,
    entry_point: &str,
) -> std::fmt::Result {
    writeln!(
        s,
        r#"
        OpCapability Shader
   %1 = OpExtInstImport "GLSL.std.450"
        OpMemoryModel Logical GLSL450
        OpEntryPoint GLCompute %main "main"
        OpExecutionMode %main LocalSize {work_group_size} 1 1
        OpSource GLSL 460
        OpName %main "{entry_point}"
    "#,
        work_group_size = info.work_group_size
    )?;

    let mut types = HashSet::new();
    for var in &ir.vars {
        assemble_type(s, &mut types, var.ty)?;
    }
    assemble_type(s, &mut types, &VarType::Void)?;
    assemble_type(s, &mut types, &VarType::U32)?;

    let mut buffer_bindings = HashSet::new();
    let mut sampler_bindings = HashSet::new();
    let mut accel_bindings = false;
    assemble_bindings(
        s,
        ir,
        &mut buffer_bindings,
        &mut sampler_bindings,
        &mut accel_bindings,
    )?;

    writeln!(
        s,
        r#"
   %3 = OpTypeFunction %void
%main = OpFunction %void None %3
   %5 = OpLabel
    "#
    )?;

    assemble_vars(s, ir)?;

    writeln!(
        s,
        r#"
        OpReturn
        OpFunctionEnd
    "#
    )?;

    Ok(())
}

pub fn assemble_vars(s: &mut String, ir: &IR) -> std::fmt::Result {
    for id in ir.var_ids() {
        let var = ir.var(id);
        let deps = ir.deps(id);
        let dst = Reg(id);
        let spv_ty = SpvType(var.ty);

        match var.op {
            crate::op::KernelOp::Nop => todo!(),
            crate::op::KernelOp::Scatter(_) => {
                let dst = deps[0];
                let src = deps[1];
                let idx = deps[2];

                let buffer_idx = ir.var(src).data;

                let dst = Reg(dst);

                writeln!(s, "{dst}_buffer_idx = OpConstant %u32 {buffer_idx}")?;
                writeln!(s, "{dst}_ptr_ty = OpTypePointer StorageBuffer %{spv_ty}")?;
                writeln!(
                    s,
                    "{dst}_ptr = OpAccessChain {dst}_ptr_ty %_var_StorageBuffer_{spv_ty} {dst}_buffer_idx"
                )?;
            }
            crate::op::KernelOp::Gather => {
                let src = deps[0];
                let idx = deps[1];
                let buffer_idx = ir.var(src).data;

                writeln!(s, "{dst}_buffer_idx = OpConstant %u32 {buffer_idx}")?;
                writeln!(s, "{dst}_ptr_ty = OpTypePointer StorageBuffer %{spv_ty}")?;
                writeln!(
                    s,
                    "{dst}_ptr = OpAccessChain {dst}_ptr_ty %_var_StorageBuffer_{spv_ty} {dst}_buffer_idx"
                )?;
            }
            crate::op::KernelOp::Index => {
                writeln!(s, "")?;
            }
            crate::op::KernelOp::Literal => {
                writeln!(s, "")?;
            }
            crate::op::KernelOp::Extract(_) => todo!(),
            crate::op::KernelOp::Construct => todo!(),
            crate::op::KernelOp::Select => todo!(),
            crate::op::KernelOp::TexLookup => todo!(),
            crate::op::KernelOp::TraceRay => todo!(),
            crate::op::KernelOp::Bop(op) => {
                let lhs = Reg(deps[0]);
                let rhs = Reg(deps[1]);
                match op {
                    crate::op::Bop::Add => writeln!(s, "")?,
                    crate::op::Bop::Sub => todo!(),
                    crate::op::Bop::Mul => todo!(),
                    crate::op::Bop::Div => todo!(),
                    crate::op::Bop::Modulus => todo!(),
                    crate::op::Bop::Min => todo!(),
                    crate::op::Bop::Max => todo!(),
                    crate::op::Bop::And => todo!(),
                    crate::op::Bop::Or => todo!(),
                    crate::op::Bop::Xor => todo!(),
                    crate::op::Bop::Shl => todo!(),
                    crate::op::Bop::Shr => todo!(),
                    crate::op::Bop::Eq => todo!(),
                    crate::op::Bop::Neq => todo!(),
                    crate::op::Bop::Lt => todo!(),
                    crate::op::Bop::Le => todo!(),
                    crate::op::Bop::Gt => todo!(),
                    crate::op::Bop::Ge => todo!(),
                };
            }
            crate::op::KernelOp::Uop(_) => todo!(),
            _ => {}
        }
    }
    Ok(())
}

pub fn assemble_buffer_binding(
    s: &mut String,
    ir: &IR,
    buffer_bindings: &mut HashSet<&'static VarType>,
    ty: &'static VarType,
) -> std::fmt::Result {
    if !buffer_bindings.contains(ty) {
        let spv_ty = SpvType(ty);
        writeln!(s, "%_runtimearr_{spv_ty} = OpTypeRuntimeArray %{spv_ty}")?;
        writeln!(
            s,
            "%_struct_runtimearr_{spv_ty} = OpTypeStruct %_runtimearr_{spv_ty}"
        )?;
        writeln!(s, "%_ptr_StorageBuffer_{spv_ty} = OpTypePointer StorageBuffer %_struct_runtimearr_{spv_ty}")?;
        writeln!(
            s,
            "%_var_StorageBuffer_{spv_ty} = OpVariable %_ptr_StorageBuffer_{spv_ty} StorageBuffer"
        )?;
        buffer_bindings.insert(ty);
    }
    Ok(())
}

pub fn assemble_bindings(
    s: &mut String,
    ir: &IR,
    buffer_bindings: &mut HashSet<&'static VarType>,
    sampler_bindings: &mut HashSet<usize>,
    accel_bindings: &mut bool,
) -> std::fmt::Result {
    for id in ir.var_ids() {
        let var = ir.var(id);
        match var.op {
            crate::op::KernelOp::BufferRef => {
                assemble_buffer_binding(s, ir, buffer_bindings, var.ty)?;
            }
            crate::op::KernelOp::TextureRef { dim } => todo!(),
            crate::op::KernelOp::AccelRef => todo!(),
            _ => {}
        }
    }
    Ok(())
}

pub fn assemble_type(
    s: &mut String,
    types: &mut HashSet<&'static VarType>,
    ty: &'static VarType,
) -> std::fmt::Result {
    if !types.contains(ty) {
        let ty_name = SpvType(ty);
        write!(s, "%{ty_name} = ")?;
        match ty {
            VarType::Void => writeln!(s, "OpTypeVoid")?,
            VarType::Bool => writeln!(s, "OpTypeBool")?,
            VarType::I8 => writeln!(s, "OpTypeInt 8 1")?,
            VarType::U8 => writeln!(s, "OpTypeInt 8 0")?,
            VarType::I16 => writeln!(s, "OpTypeInt 16 1")?,
            VarType::U16 => writeln!(s, "OpTypeInt 16 0")?,
            VarType::I32 => writeln!(s, "OpTypeInt 32 1")?,
            VarType::U32 => writeln!(s, "OpTypeInt 32 0")?,
            VarType::I64 => writeln!(s, "OpTypeInt 64 1")?,
            VarType::U64 => writeln!(s, "OpTypeInt 64 0")?,
            VarType::F16 => writeln!(s, "OpTypeFloat 16")?,
            VarType::F32 => writeln!(s, "OpTypeFloat 32")?,
            VarType::F64 => writeln!(s, "OpTypeFloat 64")?,
            VarType::Vec { ty, num } => writeln!(s, "OpTypeVector %{ty} {num}", ty = SpvType(ty))?,
            VarType::Array { ty, num } => {
                todo!();
            }
            VarType::Mat { ty, rows, cols } => todo!(),
            VarType::Struct { tys } => todo!(),
        };
        types.insert(ty);
    }
    Ok(())
}
