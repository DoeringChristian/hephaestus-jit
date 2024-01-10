use super::glslext::GLSL450Instruction;
use super::SpirvBuilder;
use crate::ir::{VarId, IR};
use crate::op::{Bop, KernelOp, ReduceOp, Uop};
use crate::vartype::{self, AsVarType, Intersection, VarType};
use rspirv::{dr, spirv};

macro_rules! glsl_ext {
    ($self:ident, $ty:ident, $ext:ident, $($operand:expr),*) => {
        {
        let ext = $self.glsl_ext;
        $self.ext_inst(
            $ty,
            None,
            ext,
            GLSL450Instruction::$ext as _,
            [$(dr::Operand::IdRef($operand)),*],
        )
        }
    };
}

impl SpirvBuilder {
    pub fn bop(
        &mut self,
        op: Bop,
        result_type: &'static VarType,
        lhs_type: &'static VarType,
        rhs_type: &'static VarType,
        lhs: u32,
        rhs: u32,
    ) -> Result<u32, dr::Error> {
        let spv_type = self.spirv_ty(result_type);
        match (lhs_type, rhs_type) {
            (VarType::Bool, VarType::Bool) => self.bool_bop(op, spv_type, lhs, rhs),
            (
                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
            ) => self.u_bop(op, spv_type, lhs, rhs),
            (
                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
            ) => self.s_bop(op, spv_type, lhs, rhs),
            (
                VarType::F16 | VarType::F32 | VarType::F64,
                VarType::F16 | VarType::F32 | VarType::F64,
            ) => self.f_bop(op, spv_type, lhs, rhs),
            (VarType::Vec { ty: lhs_type, .. }, VarType::Vec { ty: rhs_type, .. }) => {
                self.bop(op, result_type, lhs_type, rhs_type, lhs, rhs)
            }
            _ => todo!(),
        }
    }
    fn bool_bop(
        &mut self,
        op: Bop,
        result_type: u32,
        lhs: u32,
        rhs: u32,
    ) -> Result<u32, dr::Error> {
        match op {
            Bop::And => self.logical_and(result_type, None, lhs, rhs),
            Bop::Or => self.logical_or(result_type, None, lhs, rhs),
            Bop::Xor => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                let n_rhs = self.logical_not(result_type, None, rhs)?;
                let t0 = self.logical_and(result_type, None, n_lhs, rhs)?;
                let t1 = self.logical_and(result_type, None, lhs, n_rhs)?;
                self.logical_or(result_type, None, t0, t1)
            }
            Bop::Eq => self.logical_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.logical_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                self.logical_and(result_type, None, n_lhs, rhs)
            }
            Bop::Le => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                let n_rhs = self.logical_not(result_type, None, rhs)?;
                let t0 = self.logical_and(result_type, None, n_lhs, rhs)?;
                let t1 = self.logical_and(result_type, None, lhs, rhs)?;
                let t2 = self.logical_and(result_type, None, n_lhs, n_rhs)?;

                let t3 = self.logical_or(result_type, None, t0, t1)?;
                self.logical_or(result_type, None, t3, t2)
            }
            Bop::Gt => {
                let n_rhs = self.logical_not(result_type, None, lhs)?;
                self.logical_and(result_type, None, lhs, n_rhs)
            }
            Bop::Ge => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                let n_rhs = self.logical_not(result_type, None, rhs)?;
                let t0 = self.logical_and(result_type, None, lhs, n_rhs)?;
                let t1 = self.logical_and(result_type, None, lhs, rhs)?;
                let t2 = self.logical_and(result_type, None, n_lhs, n_rhs)?;

                let t3 = self.logical_or(result_type, None, t0, t1)?;
                self.logical_or(result_type, None, t3, t2)
            }
            _ => todo!(),
        }
    }
    fn u_bop(&mut self, op: Bop, result_type: u32, lhs: u32, rhs: u32) -> Result<u32, dr::Error> {
        match op {
            Bop::Add => self.i_add(result_type, None, lhs, rhs),
            Bop::Sub => self.i_sub(result_type, None, lhs, rhs),
            Bop::Mul => self.i_mul(result_type, None, lhs, rhs),
            Bop::Div => self.u_div(result_type, None, lhs, rhs),
            Bop::Modulus => self.u_mod(result_type, None, lhs, rhs),
            Bop::Min => glsl_ext!(self, result_type, UMin, lhs, rhs),
            Bop::Max => glsl_ext!(self, result_type, UMax, lhs, rhs),
            Bop::And => self.bitwise_and(result_type, None, lhs, rhs),
            Bop::Or => self.bitwise_or(result_type, None, lhs, rhs),
            Bop::Xor => self.bitwise_xor(result_type, None, lhs, rhs),
            Bop::Shl => self.shift_left_logical(result_type, None, lhs, rhs),
            Bop::Shr => self.shift_right_logical(result_type, None, lhs, rhs),
            Bop::Eq => self.i_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.i_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => self.u_less_than(result_type, None, lhs, rhs),
            Bop::Le => self.u_less_than_equal(result_type, None, lhs, rhs),
            Bop::Gt => self.u_greater_than(result_type, None, lhs, rhs),
            Bop::Ge => self.u_greater_than_equal(result_type, None, lhs, rhs),
        }
    }
    fn s_bop(&mut self, op: Bop, result_type: u32, lhs: u32, rhs: u32) -> Result<u32, dr::Error> {
        match op {
            Bop::Add => self.i_add(result_type, None, lhs, rhs),
            Bop::Sub => self.i_sub(result_type, None, lhs, rhs),
            Bop::Mul => self.i_mul(result_type, None, lhs, rhs),
            Bop::Div => self.s_div(result_type, None, lhs, rhs),
            Bop::Modulus => self.s_mod(result_type, None, lhs, rhs),
            Bop::Min => glsl_ext!(self, result_type, SMin, lhs, rhs),
            Bop::Max => glsl_ext!(self, result_type, SMax, lhs, rhs),
            Bop::And => self.bitwise_and(result_type, None, lhs, rhs),
            Bop::Or => self.bitwise_or(result_type, None, lhs, rhs),
            Bop::Xor => self.bitwise_xor(result_type, None, lhs, rhs),
            Bop::Shl => self.shift_left_logical(result_type, None, lhs, rhs),
            Bop::Shr => self.shift_right_logical(result_type, None, lhs, rhs),
            Bop::Eq => self.i_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.i_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => self.s_less_than(result_type, None, lhs, rhs),
            Bop::Le => self.s_less_than_equal(result_type, None, lhs, rhs),
            Bop::Gt => self.s_greater_than(result_type, None, lhs, rhs),
            Bop::Ge => self.s_greater_than_equal(result_type, None, lhs, rhs),
        }
    }
    fn f_bop(&mut self, op: Bop, result_type: u32, lhs: u32, rhs: u32) -> Result<u32, dr::Error> {
        match op {
            Bop::Add => self.f_add(result_type, None, lhs, rhs),
            Bop::Sub => self.f_sub(result_type, None, lhs, rhs),
            Bop::Mul => self.f_mul(result_type, None, lhs, rhs),
            Bop::Div => self.f_div(result_type, None, lhs, rhs),
            Bop::Modulus => self.f_mod(result_type, None, lhs, rhs),
            Bop::Min => glsl_ext!(self, result_type, FMin, lhs, rhs),
            Bop::Max => glsl_ext!(self, result_type, FMax, lhs, rhs),
            Bop::Eq => self.f_ord_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.f_ord_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => self.f_ord_less_than(result_type, None, lhs, rhs),
            Bop::Le => self.f_ord_less_than_equal(result_type, None, lhs, rhs),
            Bop::Gt => self.f_ord_greater_than(result_type, None, lhs, rhs),
            Bop::Ge => self.f_ord_less_than_equal(result_type, None, lhs, rhs),
            _ => todo!(),
        }
    }
}
