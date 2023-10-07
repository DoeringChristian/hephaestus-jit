use crate::trace::*;

use super::param_layout::ParamLayout;
pub fn prefix(ty: &VarType) -> &'static str {
    match ty {
        VarType::Void => "%u",
        VarType::Bool => "%p",
        VarType::I8 => "%b",
        VarType::U8 => "%b",
        VarType::I16 => "%w",
        VarType::U16 => "%w",
        VarType::I32 => "%r",
        VarType::U32 => "%r",
        VarType::I64 => "%rd",
        VarType::U64 => "%rd",
        VarType::F16 => "%h",
        VarType::F32 => "%f",
        VarType::F64 => "%d",
        _ => todo!(),
    }
}
// Retuns the cuda/ptx Representation for this type
pub fn tyname(ty: &VarType) -> &'static str {
    match ty {
        VarType::Void => "???",
        VarType::Bool => "pred",
        VarType::I8 => "s8",
        VarType::U8 => "u8",
        VarType::I16 => "s16",
        VarType::U16 => "u16",
        VarType::I32 => "s32",
        VarType::U32 => "u32",
        VarType::I64 => "s64",
        VarType::U64 => "u64",
        VarType::F16 => "f16",
        VarType::F32 => "f32",
        VarType::F64 => "f64",
        _ => todo!(),
    }
}
pub fn tyname_bin(ty: &VarType) -> &'static str {
    match ty {
        VarType::Void => "???",
        VarType::Bool => "pred",
        VarType::I8 => "b8",
        VarType::U8 => "b8",
        VarType::I16 => "b16",
        VarType::U16 => "b16",
        VarType::I32 => "b32",
        VarType::U32 => "b32",
        VarType::I64 => "b64",
        VarType::U64 => "b64",
        VarType::F16 => "b16",
        VarType::F32 => "b32",
        VarType::F64 => "b64",
        _ => todo!(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Reg<'a> {
    reg: usize,
    ty: &'a VarType,
}

impl<'a> std::fmt::Display for Reg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", prefix(&self.ty), self.reg)
    }
}

pub fn assemble_trace(
    asm: &mut impl std::fmt::Write,
    trace: &Trace,
    opid: OpId,
    param_ty: &str,
    param_layout: ParamLayout,
) -> std::fmt::Result {
    let reg = |id: VarId| Reg {
        reg: id.0,
        ty: &trace.var_ty(id),
    };

    // Write out debug info:
    writeln!(asm, "// {:?}:", trace.op(opid))?;

    let op = trace.op(opid);
    match op {
        Op::Add { dst, lhs, rhs } => {
            writeln!(
                asm,
                "\tadd.{} {}, {}, {};",
                tyname(trace.var_ty(*dst)),
                reg(*dst),
                reg(*lhs),
                reg(*rhs)
            )?;
        }
        Op::Scatter { dst, src, idx } => {
            let dst_var = trace.var(*dst);

            assert!(dst_var.external.is_some());
            assert_eq!(dst_var.ty, VarType::Array);

            let param_offset = param_layout.array_offset(dst_var.external.unwrap());

            writeln!(asm, "\tld.{param_ty}.u64 %rd0 [params+{}]", param_offset)?;

            // Multiply idx with type size and add ptr
            let ty = trace.var_ty(*src);
            writeln!(
                asm,
                "\tmad.wide.{ty} %rd3, {idx}, {ty_size}, %rd0;",
                ty = tyname(ty),
                idx = reg(*idx),
                ty_size = ty.size(),
            )?;

            let op_type = "st";
            let op = "";
            writeln!(
                asm,
                "\t{}.global{}.{} [%rd3], {};",
                op_type,
                op,
                tyname(ty),
                reg(*src),
            )?;
        }
        Op::Gather { dst, src, idx } => {
            let src_var = trace.var(*src);

            assert!(src_var.external.is_some());
            assert_eq!(src_var.ty, VarType::Array);

            // Load array ptr:

            let param_offset = param_layout.array_offset(src_var.external.unwrap());

            writeln!(asm, "\tld.{param_ty}.u64 %rd0 [params+{}]", param_offset)?;

            // Multiply idx with type size and add ptr
            let ty = trace.var_ty(*dst);
            writeln!(
                asm,
                "\tmad.wide.{ty} %rd3, {idx}, {ty_size}, %rd0;",
                ty = tyname(ty),
                idx = reg(*idx),
                ty_size = ty.size(),
            )?;

            // Load value from buffer
            writeln!(
                asm,
                "\tld.global.nc.{ty} {ptr}, [%rd3];",
                ty = tyname(ty),
                ptr = reg(*dst),
            )?;
        }
    };
    Ok(())
}
