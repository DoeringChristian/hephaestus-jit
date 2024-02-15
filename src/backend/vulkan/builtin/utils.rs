use ash::vk;

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
pub fn component_type(ty: &VarType) -> vk::ComponentTypeKHR {
    match ty {
        VarType::I8 => vk::ComponentTypeKHR::SINT8,
        VarType::U8 => vk::ComponentTypeKHR::UINT8,
        VarType::I16 => vk::ComponentTypeKHR::SINT16,
        VarType::U16 => vk::ComponentTypeKHR::UINT16,
        VarType::I32 => vk::ComponentTypeKHR::SINT32,
        VarType::U32 => vk::ComponentTypeKHR::UINT32,
        VarType::I64 => vk::ComponentTypeKHR::SINT64,
        VarType::U64 => vk::ComponentTypeKHR::UINT64,
        VarType::F16 => vk::ComponentTypeKHR::FLOAT16,
        VarType::F32 => vk::ComponentTypeKHR::FLOAT32,
        VarType::F64 => vk::ComponentTypeKHR::FLOAT64,
        _ => todo!(),
    }
}
