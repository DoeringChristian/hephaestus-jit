use std::collections::HashMap;

use ash::vk;
use text_placeholder::Template;

use super::shader_cache::ShaderKind;

use crate::backend::vulkan::buffer::BufferInfo;
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
use super::{accel, VulkanDevice};

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

impl VulkanDevice {
    pub fn reduce(&self, ctx: &mut Context, op: DeviceOp, ty: &VarType) {
        let num: usize = 1024;

        let scratch_buffer = ctx.buffer(BufferInfo {
            size: todo!(),
            alignment: todo!(),
            usage: todo!(),
            memory_location: todo!(),
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

        for i in (1..((num - 1).ilog(32) + 1)).rev() {
            pipeline.submit(
                ctx.cb,
                &ctx,
                &[WriteSet {
                    set: 0,
                    binding: 0,
                    buffers: &[BufferWriteInfo {
                        buffer: &scratch_buffer,
                    }],
                }],
                (i, 1, 1),
            );

            let memory_barriers = [vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build()];
            unsafe {
                ctx.cmd_pipeline_barrier(
                    ctx.cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &memory_barriers,
                    &[],
                    &[],
                );
            }
            todo!();
        }

        todo!()
    }
    pub fn build_accel<'a>(
        &'a self,
        ctx: &mut Context,
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

        accel.build(ctx, desc);
    }
}
