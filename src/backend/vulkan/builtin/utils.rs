use crate::vartype::VarType;

pub fn glsl_ty(ty: &VarType) -> &'static str {
    match ty {
        VarType::Bool => "uint8_t", // NOTE: use uint8_t for bool
        VarType::I8 => "int8_t",
        VarType::U8 => "uint8_t",
        VarType::I16 => "int16_t",
        VarType::U16 => "uint16_t",
        VarType::I32 => "int32_t",
        VarType::U32 => "uint32_t",
        VarType::I64 => "int64_t",
        VarType::U64 => "uint64_t",
        VarType::F16 => "float16_t",
        VarType::F32 => "float32_t",
        VarType::F64 => "float64_t",
        _ => todo!(),
    }
}
// TODO: imporve this
pub fn glsl_short_ty(ty: &VarType) -> &'static str {
    match ty {
        VarType::I8 => "i8",
        VarType::U8 => "u8",
        VarType::I16 => "i16",
        VarType::U16 => "u16",
        VarType::I32 => "i32",
        VarType::U32 => "u32",
        VarType::I64 => "i64",
        VarType::U64 => "u64",
        VarType::F16 => "f16",
        VarType::F32 => "f32",
        VarType::F64 => "f64",
        _ => todo!(),
    }
}
