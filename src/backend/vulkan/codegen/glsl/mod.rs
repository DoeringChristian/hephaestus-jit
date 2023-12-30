use half::f16;

use super::CompileInfo;
use crate::ir::{VarId, IR};
use crate::vartype::{AsVarType, VarType};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

pub fn assemble_ir(ir: &IR, info: &CompileInfo, entry_point: &str) -> Option<Vec<u32>> {
    let mut s = String::new();
    assemble_entry_point(&mut s, ir, info, entry_point).unwrap();

    // log::trace!("\n{s}");
    print!("{s}");

    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    let mut compiler = shaderc::Compiler::new().unwrap();

    let artefact = compiler
        .compile_into_spirv(
            &s,
            shaderc::ShaderKind::Compute,
            "",
            entry_point,
            Some(&options),
        )
        .unwrap();

    Some(artefact.as_binary().to_vec())
}

pub fn assemble_entry_point(
    s: &mut String,
    ir: &IR,
    info: &CompileInfo,
    entry_point: &str,
) -> std::fmt::Result {
    write!(
        s,
        r#"
#version 450

// Explicit arythmetic types
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require

// Atomics and Memory
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_atomic_int64: require
        
"#
    )?;

    let mut buffer_types = HashSet::new();
    bindings(s, ir, &mut buffer_types)?;
    let ty = u32::var_ty();
    if !buffer_types.contains(ty) {
        let n_buffers = ir.n_buffers + 1;
        writeln!(
            s,
            "layout(set = 0, binding = 0) buffer Buffer_{name}{{ {name} b[]; }} buffer_{name}[{n_buffers}];",
            name = GlslTypeName(ty)
        )?;
    }

    writeln!(s, "")?;
    writeln!(
        s,
        "layout(local_size_x = {workgroup_size}, local_size_y = 1, local_size_z = 1)in;",
        workgroup_size = info.work_group_size
    )?;
    writeln!(s, "void {entry_point}(){{")?;

    // Early return
    writeln!(s, "\tuint size = buffer_uint32_t[0].b[0];")?;
    writeln!(s, "\tuint index = uint(gl_GlobalInvocationID.x);")?;
    writeln!(s, "\tif (index >= size) {{return;}}")?;

    assemble_vars(s, ir)?;

    writeln!(s, "}}")?;

    Ok(())
}

fn bindings(
    s: &mut String,
    ir: &IR,
    buffer_types: &mut HashSet<&'static VarType>,
) -> std::fmt::Result {
    let n_buffers = ir.n_buffers + 1;
    for id in ir.var_ids() {
        let var = ir.var(id);
        match var.op {
            crate::op::KernelOp::BufferRef => {
                if !buffer_types.contains(var.ty) {
                    writeln!(
                        s,
                        "layout(set = 0, binding = 0) buffer Buffer_{name}{{ {name} b[]; }} buffer_{name}[{n_buffers}];",
                        name = GlslTypeName(var.ty)
                    )?;
                    buffer_types.insert(var.ty);
                }
            }
            crate::op::KernelOp::TextureRef { dim } => todo!(),
            crate::op::KernelOp::AccelRef => todo!(),
            _ => {}
        }
    }
    Ok(())
}
pub struct GlslTypeName(&'static VarType);
impl std::fmt::Display for GlslTypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            VarType::Void => write!(f, "void"),
            VarType::Bool => write!(f, "bool"),
            VarType::I8 => write!(f, "int8_t"),
            VarType::U8 => write!(f, "uint8_t"),
            VarType::I16 => write!(f, "int16_t"),
            VarType::U16 => write!(f, "uint16_t"),
            VarType::I32 => write!(f, "int32_t"),
            VarType::U32 => write!(f, "uint32_t"),
            VarType::I64 => write!(f, "int64_t"),
            VarType::U64 => write!(f, "uint64_t"),
            VarType::F16 => write!(f, "float16_t"),
            VarType::F32 => write!(f, "float32_t"),
            VarType::F64 => write!(f, "float64_t"),
            VarType::Vec { ty, num } => match ty {
                VarType::I8 => write!(f, "i8vec{num}"),
                VarType::U8 => write!(f, "u8vec{num}"),
                VarType::I16 => write!(f, "i16{num}"),
                VarType::U16 => write!(f, "u16vec{num}"),
                VarType::I32 => write!(f, "i32vec{num}"),
                VarType::U32 => write!(f, "u32vec{num}"),
                VarType::I64 => write!(f, "i64vec{num}"),
                VarType::U64 => write!(f, "u64vec{num}"),
                VarType::F16 => write!(f, "f16vec{num}"),
                VarType::F32 => write!(f, "f32vec{num}"),
                VarType::F64 => write!(f, "f64vec{num}"),
                _ => todo!(),
            },
            VarType::Array { ty, num } => todo!(),
            VarType::Mat { ty, rows, cols } => todo!(),
            VarType::Struct { tys } => todo!(),
        }
    }
}

pub struct Reg(VarId);
impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{id}", id = self.0 .0)
    }
}

fn assemble_vars(s: &mut String, ir: &IR) -> std::fmt::Result {
    for id in ir.var_ids() {
        let var = ir.var(id);
        let reg = id.0;
        let deps = ir.deps(id);

        let ty = var.ty;
        let glsl_ty = GlslTypeName(ty);
        // let ty = GlslTypeName(var.ty);

        match var.op {
            crate::op::KernelOp::Nop => {
                writeln!(
                    s,
                    "\t{glsl_ty} {dst} = {src};",
                    dst = Reg(id),
                    src = Reg(deps[0])
                )?;
            }
            crate::op::KernelOp::Scatter(_) => {
                let dst = deps[0];
                let src = deps[1];
                let idx = deps[2];
                let glsl_ty = GlslTypeName(ir.var(src).ty);
                let buffer_idx = ir.var(dst).data + 1;

                writeln!(
                    s,
                    "\tbuffer_{glsl_ty}[{buffer_idx}].b[{idx}] = {src};",
                    idx = Reg(idx),
                    src = Reg(src)
                )?;
            }
            crate::op::KernelOp::Gather => {
                let src = deps[0];
                let idx = deps[1];
                writeln!(
                    s,
                    "\t{glsl_ty} {dst} = buffer_{glsl_ty}[{buffer_idx}].b[{idx}];",
                    dst = Reg(id),
                    buffer_idx = ir.var(src).data + 1,
                    idx = Reg(deps[1]),
                )?;
            }
            crate::op::KernelOp::Index => {
                // writeln!(
                //     s,
                //     "\tuint32_t {dst} = uint32_t(gl_GlobalInvocationID.x);",
                //     dst = Reg(id)
                // )?;
                writeln!(s, "\tuint32_t {dst} = index;", dst = Reg(id))?;
            }
            crate::op::KernelOp::Literal => {
                let dst = Reg(id);
                match ty {
                    VarType::Bool => {
                        writeln!(
                            s,
                            "\t{glsl_ty} {dst} = {glsl_ty}({data});",
                            data = if var.data == 0 { false } else { true }
                        )?;
                    }
                    VarType::I8
                    | VarType::U8
                    | VarType::I16
                    | VarType::U16
                    | VarType::I32
                    | VarType::U32
                    | VarType::I64
                    | VarType::U64 => {
                        let data = var.data;
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data}ul);",)?;
                    }
                    VarType::F16 => {
                        let data: f16 = unsafe { *(&var.data as *const _ as *const _) };
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data});",)?;
                    }
                    VarType::F32 => {
                        let data: f32 = unsafe { *(&var.data as *const _ as *const _) };
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data});",)?;
                    }
                    VarType::F64 => {
                        let data: f64 = unsafe { *(&var.data as *const _ as *const _) };
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data});",)?;
                    }
                    _ => todo!(),
                    // VarType::Vec { ty, num } => todo!(),
                    // VarType::Array { ty, num } => todo!(),
                    // VarType::Mat { ty, rows, cols } => todo!(),
                    // VarType::Struct { tys } => todo!(),
                }
            }
            crate::op::KernelOp::Extract(_) => todo!(),
            crate::op::KernelOp::Construct => todo!(),
            crate::op::KernelOp::Select => todo!(),
            crate::op::KernelOp::TexLookup => todo!(),
            crate::op::KernelOp::TraceRay => todo!(),
            crate::op::KernelOp::Bop(op) => {
                let dst = Reg(id);
                let lhs = Reg(deps[0]);
                let rhs = Reg(deps[1]);
                match op {
                    crate::op::Bop::Add => {
                        writeln!(s, "\t{glsl_ty} {dst} = {lhs} + {rhs};")?;
                    }
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
                }
            }
            crate::op::KernelOp::Uop(_) => todo!(),
            _ => {}
        }
    }
    Ok(())
}
