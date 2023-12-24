use std::collections::HashMap;

use ash::vk;
use std::sync::Arc;
use text_placeholder::Template;

use super::pool::Pool;
use super::shader_cache::ShaderKind;
use super::vulkan_core::graph::RGraph;

use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};
use crate::backend::vulkan::vulkan_core::buffer::{BufferInfo, MemoryLocation};
use crate::backend::{self, AccelDesc};
use crate::graph::{Pass, PassOp};
use crate::op::{DeviceOp, ReduceOp};
use crate::vartype::VarType;

use super::accel::Accel;
use super::vulkan_core::buffer::Buffer;
use super::{accel, VulkanDevice};

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
pub fn round_pow2(x: u32) -> u32 {
    let mut x = x;
    x -= 1;
    x |= x.overflowing_shr(1).0;
    x |= x.overflowing_shr(2).0;
    x |= x.overflowing_shr(4).0;
    x |= x.overflowing_shr(8).0;
    x |= x.overflowing_shr(16).0;
    x |= x.overflowing_shr(32).0;
    return x + 1;
}

#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub block_size: u32,
    pub grid_size: u32,
}

impl VulkanDevice {
    pub fn get_launch_config(&self, size: usize) -> LaunchConfig {
        let max_block_size = 1024;
        LaunchConfig {
            grid_size: ((size + max_block_size - 1) / max_block_size) as u32,
            block_size: max_block_size as u32,
        }
    }
    pub fn build_accel<'a>(
        &'a self,
        rgraph: &mut RGraph,
        accel_desc: &AccelDesc,
        accel: &Accel,
        buffers: impl IntoIterator<Item = &'a Arc<Buffer>>,
    ) {
        // WARN: This is potentially very unsafe, since we are just randomly
        // accessing the buffers and hoping for them to be index/vertex
        // buffers

        let mut buffers = buffers.into_iter();

        let instances = &buffers.next().unwrap();

        let geometries = accel_desc
            .geometries
            .iter()
            .map(|g| match g {
                backend::GeometryDesc::Triangles { .. } => {
                    accel::AccelGeometryBuildInfo::Triangles {
                        triangles: &buffers.next().unwrap(),
                        vertices: &buffers.next().unwrap(),
                    }
                }
            })
            .collect::<Vec<_>>();

        let desc = accel::AccelBuildInfo {
            geometries: &geometries,
            instances,
        };

        accel.build(rgraph, desc);
    }
}
