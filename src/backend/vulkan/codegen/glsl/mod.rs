use super::CompileInfo;
use crate::ir::{VarId, IR};
use crate::vartype::VarType;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

pub fn assemble_trace(ir: &IR, info: &CompileInfo, entry_point: &str) -> Option<Vec<u32>> {
    todo!()
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

    writeln!(s, "void main(){{")?;

    writeln!(s, "}}")?;

    todo!()
}

fn bindings(s: &mut String, ir: &IR) -> std::fmt::Result {
    let n_buffers = ir.n_buffers + 1;
    let mut buffer_types = HashSet::new();
    for id in ir.var_ids() {
        let var = ir.var(id);
        match var.op {
            crate::op::KernelOp::BufferRef => {
                if !buffer_types.contains(var.ty) {
                    let buffer_name = type_name_buffer(var.ty);
                    writeln!(
                        s,
                        "layout(set = 0, binding = 0) buffer Buffer{}",
                        buffer_name
                    )?;
                    buffer_types.insert(var.ty);
                }
            }
            crate::op::KernelOp::TextureRef { dim } => todo!(),
            crate::op::KernelOp::AccelRef => todo!(),
            _ => todo!(),
        }
    }
    Ok(())
}

fn type_name_buffer(ty: &VarType) -> String {
    let ty_name = match ty {
        VarType::Bool => format!("Bool"),
        VarType::I8 => format!("I8"),
        VarType::U8 => format!("U8"),
        VarType::I16 => format!("I16"),
        VarType::U16 => format!("U16"),
        VarType::I32 => format!("I32"),
        VarType::U32 => format!("U32"),
        VarType::I64 => format!("I64"),
        VarType::U64 => format!("U64"),
        VarType::F16 => format!("F16"),
        VarType::F32 => format!("F32"),
        VarType::F64 => format!("F64"),
        VarType::Vec { ty, num } => format!("Vec{num}{ty}", ty = type_name_buffer(ty)),
        VarType::Array { ty, num } => format!("Arr{num}{ty}", ty = type_name_buffer(ty)),
        VarType::Mat { ty, rows, cols } => {
            format!("Mat{rows}x{cols}{ty}", ty = type_name_buffer(ty))
        }
        VarType::Struct { tys } => format!(
            "StructStart{}StructEnd",
            tys.iter()
                .map(|ty| type_name_buffer(ty))
                .reduce(|a, b| format!("{a}{b}"))
                .unwrap()
        ),
        _ => todo!(),
    };
    ty_name
}

fn assemble_vars(asm: &mut String, ir: &IR, info: &CompileInfo, entry_point: &str) {}
