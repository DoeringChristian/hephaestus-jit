use std::collections::HashMap;

use ash::vk;
use text_placeholder::Template;

use super::shader_cache::ShaderKind;

use crate::backend::vulkan::buffer::{BufferInfo, MemoryLocation};
use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};
use crate::backend::{self, AccelDesc};
use crate::graph::{Pass, PassOp};
use crate::op::DeviceOp;
use crate::vartype::VarType;

use super::accel::Accel;
use super::buffer::Buffer;
use super::context::Context;
use super::{accel, rgraph, VulkanDevice};

pub fn glsl_ty(ty: &VarType) -> &'static str {
    match ty {
        // VarType::Bool => "bool",
        VarType::I8 => "uint8_t",
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

impl VulkanDevice {
    pub fn reduce(
        &self,
        rgraph: &mut rgraph::RGraph,
        op: DeviceOp,
        ty: &VarType,
        num: usize,
        src: &Buffer,
        dst: &Buffer,
    ) {
        let ty_size = ty.size();

        let scratch_buffer = rgraph.resource(BufferInfo {
            size: round_pow2((num * ty.size()) as u32) as usize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        });

        let reduction = match op {
            DeviceOp::Max => "max(a, b)",
            _ => todo!(),
        };
        let ty = glsl_ty(ty);

        let template = Template::new(include_str!("kernels/reduce.glsl"));
        let defines = HashMap::from([("REDUCE", reduction), ("TYPE", ty)]);

        let shader =
            self.get_shader_glsl(&template.fill_with_hashmap(&defines), ShaderKind::Compute);
        let pipeline = self.get_pipeline(&PipelineDesc {
            code: &shader,
            desc_set_layouts: &[DescSetLayout {
                bindings: &[Binding {
                    binding: 0,
                    count: 1,
                }],
            }],
        });

        rgraph.pass().exec(|rgraph, cb| {
            unsafe {
                rgraph.cmd_copy_buffer(
                    cb,
                    src.buffer(),
                    rgraph.buffer(scratch_buffer).buffer(),
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: vk::WHOLE_SIZE,
                    }],
                )
            };
        });

        // let memory_barriers = [vk::MemoryBarrier::builder()
        //     .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        //     .dst_access_mask(vk::AccessFlags::SHADER_READ)
        //     .build()];
        // unsafe {
        //     ctx.cmd_pipeline_barrier(
        //         ctx.cb,
        //         vk::PipelineStageFlags::ALL_COMMANDS,
        //         vk::PipelineStageFlags::ALL_COMMANDS,
        //         vk::DependencyFlags::empty(),
        //         &memory_barriers,
        //         &[],
        //         &[],
        //     );
        // }

        for i in (1..((num - 1).ilog(32) + 1)).rev() {
            rgraph.pass().exec(|rgraph, cb| {
                pipeline.submit(
                    cb,
                    rgraph,
                    &[WriteSet {
                        set: 0,
                        binding: 0,
                        buffers: &[BufferWriteInfo {
                            buffer: &rgraph.buffer(scratch_buffer),
                        }],
                    }],
                    (i, 1, 1),
                );
            });

            // let memory_barriers = [vk::MemoryBarrier::builder()
            //     .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            //     .dst_access_mask(vk::AccessFlags::SHADER_READ)
            //     .build()];
            // unsafe {
            //     ctx.cmd_pipeline_barrier(
            //         ctx.cb,
            //         vk::PipelineStageFlags::ALL_COMMANDS,
            //         vk::PipelineStageFlags::ALL_COMMANDS,
            //         vk::DependencyFlags::empty(),
            //         &memory_barriers,
            //         &[],
            //         &[],
            //     );
            // }
        }

        rgraph.pass().exec(|rgraph, cb| {
            unsafe {
                rgraph.cmd_copy_buffer(
                    cb,
                    rgraph.buffer(scratch_buffer).buffer(),
                    dst.buffer(),
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: ty_size as _,
                    }],
                )
            };
        });
    }
    pub fn build_accel<'a>(
        &'a self,
        rgraph: &mut rgraph::RGraph,
        accel_desc: &AccelDesc,
        accel: &Accel,
        buffers: impl IntoIterator<Item = &'a Buffer>,
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
