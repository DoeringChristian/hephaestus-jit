use ash::vk;

use crate::vartype;

pub fn channels_ty_to_format(channels: usize, ty: &vartype::VarType) -> vk::Format {
    match ty {
        vartype::VarType::I8 => match channels {
            1 => vk::Format::R8_SINT,
            2 => vk::Format::R8G8_SINT,
            3 => vk::Format::R8G8B8_SINT,
            4 => vk::Format::R8G8B8A8_SINT,
            _ => todo!(),
        },
        vartype::VarType::U8 => match channels {
            1 => vk::Format::R8_UINT,
            2 => vk::Format::R8G8_UINT,
            3 => vk::Format::R8G8B8_UINT,
            4 => vk::Format::R8G8B8A8_UINT,
            _ => todo!(),
        },
        vartype::VarType::I16 => match channels {
            1 => vk::Format::R16_SINT,
            2 => vk::Format::R16G16_SINT,
            3 => vk::Format::R16G16B16_SINT,
            4 => vk::Format::R16G16B16A16_SINT,
            _ => todo!(),
        },
        vartype::VarType::U16 => match channels {
            1 => vk::Format::R16_UINT,
            2 => vk::Format::R16G16_UINT,
            3 => vk::Format::R16G16B16_UINT,
            4 => vk::Format::R16G16B16A16_UINT,
            _ => todo!(),
        },
        vartype::VarType::I32 => match channels {
            1 => vk::Format::R32_SINT,
            2 => vk::Format::R32G32_SINT,
            3 => vk::Format::R32G32B32_SINT,
            4 => vk::Format::R32G32B32A32_SINT,
            _ => todo!(),
        },
        vartype::VarType::U32 => match channels {
            1 => vk::Format::R32_UINT,
            2 => vk::Format::R32G32_UINT,
            3 => vk::Format::R32G32B32_UINT,
            4 => vk::Format::R32G32B32A32_UINT,
            _ => todo!(),
        },
        vartype::VarType::I64 => match channels {
            1 => vk::Format::R64_SINT,
            2 => vk::Format::R64G64_SINT,
            3 => vk::Format::R64G64B64_SINT,
            4 => vk::Format::R64G64B64A64_SINT,
            _ => todo!(),
        },
        vartype::VarType::U64 => match channels {
            1 => vk::Format::R64_UINT,
            2 => vk::Format::R64G64_UINT,
            3 => vk::Format::R64G64B64_UINT,
            4 => vk::Format::R64G64B64A64_UINT,
            _ => todo!(),
        },

        vartype::VarType::F16 => match channels {
            1 => vk::Format::R16_SFLOAT,
            2 => vk::Format::R16G16_SFLOAT,
            3 => vk::Format::R16G16B16_SFLOAT,
            4 => vk::Format::R16G16B16A16_SFLOAT,
            _ => todo!(),
        },
        vartype::VarType::F32 => match channels {
            1 => vk::Format::R32_SFLOAT,
            2 => vk::Format::R32G32_SFLOAT,
            3 => vk::Format::R32G32B32_SFLOAT,
            4 => vk::Format::R32G32B32A32_SFLOAT,
            _ => todo!(),
        },
        vartype::VarType::F64 => match channels {
            1 => vk::Format::R64_SFLOAT,
            2 => vk::Format::R64G64_SFLOAT,
            3 => vk::Format::R64G64B64_SFLOAT,
            4 => vk::Format::R64G64B64A64_SFLOAT,
            _ => todo!(),
        },
        _ => todo!(),
    }
}
