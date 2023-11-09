use crate::ir::*;
use crate::op::Op;
use crate::vartype::VarType;

/// Offset for the special registers:
///
/// Special registers:
///   %r0   :  Index
///   %r1   :  Step
///   %r2   :  Size
///   %p0   :  Stopping predicate
///   %rd0  :  Temporary for parameter pointers
///   %rd1  :  Pointer to parameter table in global memory if too big
///   %b3, %w3, %r3, %rd3, %f3, %d3, %p3: reserved for use in compound
///   statements that must write a temporary result to a register.
pub const REGISTER_OFFSET: usize = 4;

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
        // VarType::F16 => "%h",
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
        // VarType::F16 => "f16",
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
        // VarType::F16 => "b16",
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

impl<'a> Reg<'a> {
    /// Create a register constructor, which can be called to generate the `Reg` Representation,
    /// without having to use the trace.
    ///
    /// * `trace`: Trace to generate code for
    pub fn constructor(trace: &'a IR) -> impl Fn(VarId) -> Reg<'a> {
        move |id: VarId| Reg {
            reg: id.0 + REGISTER_OFFSET,
            ty: &trace.var_ty(id),
        }
    }
}

impl<'a> std::fmt::Display for Reg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", prefix(&self.ty), self.reg)
    }
}

pub fn assemble_trace(
    asm: &mut impl std::fmt::Write,
    trace: &IR,
    entry_point: &str,
    param_ty: &str,
    // param_layout: &ParamLayout,
) -> std::fmt::Result {
    let param_layout = ParamLayout::generate(trace);
    let n_regs = trace.vars.len() + REGISTER_OFFSET; // Add 4 utility registers

    // Generate Code
    writeln!(asm, ".version {}.{}", 8, 0)?;
    writeln!(asm, ".target {}", "sm_86")?;
    writeln!(asm, ".address_size 64")?;

    writeln!(asm, "")?;

    writeln!(asm, ".entry {}(", entry_point)?;
    writeln!(
        asm,
        "\t.param .align 8 .b8 params[{param_size}]) {{",
        param_size = param_layout.buffer_size(),
    )?;
    writeln!(asm, "")?;

    writeln!(
        asm,
        "\t.reg.b8   %b <{n_regs}>; .reg.b16 %w<{n_regs}>; .reg.b32 %r<{n_regs}>;"
    )?;
    writeln!(
        asm,
        "\t.reg.b64  %rd<{n_regs}>; .reg.f32 %f<{n_regs}>; .reg.f64 %d<{n_regs}>;"
    )?;
    writeln!(asm, "\t.reg.pred %p <{n_regs}>;")?;
    writeln!(asm, "")?;

    write!(
        asm,
        "\tmov.u32 %r0, %ctaid.x;\n\
            \tmov.u32 %r1, %ntid.x;\n\
            \tmov.u32 %r2, %tid.x;\n\
            \tmad.lo.u32 %r0, %r0, %r1, %r2; // r0 <- Index\n"
    )?;

    writeln!(asm, "")?;

    writeln!(
        asm,
        "\t// Index Conditional (jump to done if Index >= Size)."
    )?;
    writeln!(
        asm,
        "\tld.param.u32 %r2, [params]; // r2 <- params[0] (Size)"
    )?;
    writeln!(
        asm,
        "\tld.param.u64 %rd1, [params+8]; // rd1 <- params[8] (parameter table)"
    )?;

    write!(
        asm,
        "\tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r0 >= r2\n\
        \t@%p0 bra done; // if p0 => done\n\
        \t\n\
        \tmov.u32 %r3, %nctaid.x; // r3 <- nctaid.x\n\
        \tmul.lo.u32 %r1, %r3, %r1; // r1 <- r3 * r1\n\
        \t\n"
    )?;

    write!(asm, "body: // sm_{}\n", 86)?; // TODO: compute capability from device

    for var in trace.var_ids() {
        assemble_var(asm, trace, var, param_ty, &param_layout)?;
    }

    // End of kernel:

    writeln!(asm, "\n\t//End of Kernel:")?;
    writeln!(
        asm,
        "\n\tadd.u32 %r0, %r0, %r1; // r0 <- r0 + r1\n\
           \tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r1 >= r2\n\
           \t@!%p0 bra body; // if p0 => body\n\
           \n"
    )?;
    writeln!(asm, "done:")?;
    write!(
        asm,
        "\n\tret;\n\
       }}\n"
    )?;
    Ok(())
}

pub fn assemble_var(
    asm: &mut impl std::fmt::Write,
    trace: &IR,
    varid: VarId,
    param_ty: &str,
    param_layout: &ParamLayout,
) -> std::fmt::Result {
    let reg = Reg::constructor(trace);
    let deps = trace.deps(varid);

    // Write out debug info:
    writeln!(asm, "")?;

    let op = &trace.var(varid).op;
    writeln!(asm, "// {:?}:", op)?;
    match &op {
        Op::Nop => {}
        Op::Bop(bop) => {
            let lhs = deps[0];
            let rhs = deps[1];
            match bop {
                crate::op::Bop::Add => {
                    writeln!(
                        asm,
                        "\tadd.{} {}, {}, {};",
                        tyname(trace.var_ty(varid)),
                        reg(varid),
                        reg(lhs),
                        reg(rhs)
                    )?;
                }
            }
        }
        Op::Scatter => {
            let dst = deps[0];
            let src = deps[1];
            let idx = deps[2];

            let dst_var = trace.var(dst);

            // assert_eq!(dst_var.ty, VarType::Array);

            let param_offset = param_layout.byte_offset(dst);

            writeln!(
                asm,
                "\tld.global.u64 %rd0, [%rd1+{param_offset}]; //Load buffer pointer from table"
            )?;

            // Multiply idx with type size and add ptr
            let ty = trace.var_ty(src);
            writeln!(
                asm,
                "\tmad.wide.{ty} %rd3, {idx}, {ty_size}, %rd0;",
                ty = tyname(trace.var_ty(idx)),
                idx = reg(idx),
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
                reg(src),
            )?;
        }
        Op::Gather => {
            let src = deps[0];
            let idx = deps[1];

            let src_var = trace.var(src);

            // Load array ptr:

            let param_offset = param_layout.byte_offset(src);

            writeln!(
                asm,
                "\tld.global.u64 %rd0, [%rd1+{param_offset}]; //Load buffer pointer from table"
            )?;

            // Multiply idx with type size and add ptr
            let ty = trace.var_ty(varid);
            writeln!(
                asm,
                "\tmad.wide.{ty} %rd3, {idx}, {ty_size}, %rd0;",
                ty = tyname(trace.var_ty(idx)),
                idx = reg(idx),
                ty_size = ty.size(),
            )?;

            // Load value from buffer
            writeln!(
                asm,
                "\tld.global.nc.{ty} {ptr}, [%rd3];",
                ty = tyname(ty),
                ptr = reg(varid),
            )?;
        }
        Op::Index => {
            let ty = trace.var_ty(varid);
            writeln!(asm, "\tmov.{} {}, %r0;", tyname(ty), reg(varid))?;
        }
        Op::Literal => {
            // let ty = trace.var_ty(varid);
            //
            // writeln!(
            //     asm,
            //     "\tmov.{tyname} {dst}, 0x{data:x};\n",
            //     tyname = tyname_bin(trace.var_ty(varid)),
            //     dst = reg(varid),
            //     data = data,
            // )?;
            todo!()
        }
        Op::Buffer => {}
        _ => {}
    };
    Ok(())
}
