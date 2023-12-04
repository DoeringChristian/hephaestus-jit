use std::collections::HashMap;

use ash::vk;
use text_placeholder::Template;

use super::pool::Pool;
use super::shader_cache::ShaderKind;

use crate::backend::vulkan::buffer::{BufferInfo, MemoryLocation};
use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};
use crate::backend::{self, AccelDesc};
use crate::graph::{Pass, PassOp};
use crate::op::{DeviceOp, ReduceOp};
use crate::vartype::VarType;

use super::accel::Accel;
use super::buffer::Buffer;
use super::{accel, VulkanDevice};

pub fn glsl_ty(ty: &VarType) -> &'static str {
    match ty {
        // VarType::Bool => "bool",
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
        device: &VulkanDevice,
        cb: vk::CommandBuffer,
        pool: &mut Pool,
        op: ReduceOp,
        ty: &VarType,
        num: usize,
        src: &Buffer,
        dst: &Buffer,
    ) {
        let ty_size = ty.size();

        let n_passes = (num - 1).ilog(32) + 1;
        let scratch_size = 32u32.pow(n_passes);

        let mut in_buffer = pool.buffer(BufferInfo {
            size: scratch_size as usize * ty_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        });
        let mut out_buffer = pool.buffer(BufferInfo {
            size: scratch_size as usize * ty_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        });
        let mut size_buffer = pool.buffer(BufferInfo {
            size: std::mem::size_of::<u64>(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
            ..Default::default()
        });
        size_buffer
            .mapped_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&[num as u64]));

        let reduction = match op {
            ReduceOp::Max => "max(a, b)",
            ReduceOp::Min => "min(a, b)",
            ReduceOp::Sum => "(a + b)",
            _ => todo!(),
        };
        let init = match op {
            ReduceOp::Max => match ty {
                // VarType::Bool => todo!(),
                VarType::I8 => "int8_t(-0x80)",
                VarType::U8 => "uint8_t(0)",
                VarType::I16 => "int16_t(-0x8000)",
                VarType::U16 => "uint16_t(0)",
                VarType::I32 => "int32_t(-0x80000000)",
                VarType::U32 => "uint32_t(0)",
                VarType::I64 => "int64_t(-0x8000000000000000l)",
                VarType::U64 => "uint64_t(0ul)",
                VarType::F32 => "float32_t(-1.0/0.0)",
                VarType::F64 => "float64_t(-1.0/0.0)",
                _ => todo!(),
            },
            ReduceOp::Min => match ty {
                // VarType::Bool => todo!(),
                VarType::I8 => "int8_t(0x7f)",
                VarType::U8 => "uint8_t(0xff)",
                VarType::I16 => "int16_t(0x7fff)",
                VarType::U16 => "uint16_t(0xffff)",
                VarType::I32 => "int32_t(0x7fffffff)",
                VarType::U32 => "uint32_t(0xffffffff)",
                VarType::I64 => "int64_t(0x7fffffffffffffffl)",
                VarType::U64 => "uint64_t(0xfffffffffffffffful)",
                VarType::F32 => "float32_t(1.0/0.0)",
                VarType::F64 => "float64_t(1.0/0.0)",
                _ => todo!(),
            },
            ReduceOp::Sum => match ty {
                // VarType::Bool => todo!(),
                VarType::I8 => "int8_t(0)",
                VarType::U8 => "uint8_t(0)",
                VarType::I16 => "int16_t(0)",
                VarType::U16 => "uint16_t(0)",
                VarType::I32 => "int32_t(0)",
                VarType::U32 => "uint32_t(0)",
                VarType::I64 => "int64_t(0)",
                VarType::U64 => "uint64_t(0)",
                VarType::F32 => "float32_t(0)",
                VarType::F64 => "float64_t(0)",
                _ => todo!(),
            },
            _ => todo!(),
        };

        let ty = glsl_ty(ty);

        let shader = self.get_static_glsl_templated(
            include_str!("kernels/reduce.glsl"),
            &[
                ("REDUCE", reduction),
                ("INIT", init),
                ("TYPE", ty),
                ("WORK_GROUP_SIZE", "32"),
            ],
            ShaderKind::Compute,
        );
        let pipeline = self.get_pipeline(&PipelineDesc {
            code: &shader,
            desc_set_layouts: &[DescSetLayout {
                bindings: &[
                    Binding {
                        binding: 0,
                        count: 1,
                    },
                    Binding {
                        binding: 1,
                        count: 1,
                    },
                    Binding {
                        binding: 2,
                        count: 1,
                    },
                ],
            }],
        });
        log::trace!("Created templated pipeline.");

        unsafe {
            device.cmd_copy_buffer(
                cb,
                src.buffer(),
                in_buffer.buffer(),
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: src.info().size as _,
                }],
            )
        };

        let memory_barriers = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build()];
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &memory_barriers,
                &[],
                &[],
            );
        }

        log::trace!("Reducing {num} elements with {reduction}");
        log::trace!("Launching {n_passes} shader passes");
        log::trace!("Scratch Buffer size: {scratch_size}");
        for i in (0..n_passes).rev() {
            log::trace!("Launching shader");
            pipeline.submit(
                cb,
                pool,
                &device,
                &[
                    WriteSet {
                        set: 0,
                        binding: 0,
                        buffers: &[BufferWriteInfo { buffer: &in_buffer }],
                    },
                    WriteSet {
                        set: 0,
                        binding: 1,
                        buffers: &[BufferWriteInfo {
                            buffer: &out_buffer,
                        }],
                    },
                    WriteSet {
                        set: 0,
                        binding: 2,
                        buffers: &[BufferWriteInfo {
                            buffer: &size_buffer,
                        }],
                    },
                ],
                (32u32.pow(i), 1, 1),
            );

            let memory_barriers = [vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build()];
            unsafe {
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &memory_barriers,
                    &[],
                    &[],
                );
            }
            // Swap In/Out buffers
            std::mem::swap(&mut in_buffer, &mut out_buffer);
        }

        unsafe {
            device.cmd_copy_buffer(
                cb,
                in_buffer.buffer(),
                dst.buffer(),
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: ty_size as _,
                }],
            )
        };
    }
    pub fn build_accel<'a>(
        &'a self,
        cb: vk::CommandBuffer,
        pool: &mut Pool,
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

        accel.build(cb, pool, desc);
    }
}
