use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use super::super::vulkan_core::{
    buffer::{Buffer, BufferInfo},
    graph::RGraph,
};
use super::utils::*;
use crate::backend::vulkan::VulkanDevice;
use crate::backend::vulkan::{
    pipeline::{Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet},
    vkdevice::LaunchConfig,
};
use crate::utils;
use crate::vartype::VarType;
use crate::{backend::vulkan::shader_cache::ShaderKind, op::ReduceOp};

pub fn reduce(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    op: ReduceOp,
    ty: &VarType,
    num: usize,
    src: &Arc<Buffer>,
    dst: &Arc<Buffer>,
) {
    let ty_size = ty.size();

    let n_passes = (num - 1).ilog(32) + 1;
    let scratch_size = 32u32.pow(n_passes);

    let mut in_buffer = Arc::new(Buffer::create(
        device,
        BufferInfo {
            size: scratch_size as usize * ty_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        },
    ));
    let mut out_buffer = Arc::new(Buffer::create(
        device,
        BufferInfo {
            size: scratch_size as usize * ty_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        },
    ));
    let mut size_buffer = Buffer::create(
        device,
        BufferInfo {
            size: std::mem::size_of::<u64>(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
            ..Default::default()
        },
    );
    size_buffer
        .mapped_slice_mut()
        .copy_from_slice(bytemuck::cast_slice(&[num as u64]));
    let size_buffer = Arc::new(size_buffer);

    let reduction = match op {
        ReduceOp::Max => "max(a, b)",
        ReduceOp::Min => "min(a, b)",
        ReduceOp::Sum => "(a + b)",
        ReduceOp::Prod => "(a * b)",
        ReduceOp::And => "(a & b)",
        ReduceOp::Or => "(a | b)",
        ReduceOp::Xor => "(a ^ b)",
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
        ReduceOp::Prod => match ty {
            // VarType::Bool => todo!(),
            VarType::I8 => "int8_t(1)",
            VarType::U8 => "uint8_t(1)",
            VarType::I16 => "int16_t(1)",
            VarType::U16 => "uint16_t(1)",
            VarType::I32 => "int32_t(1)",
            VarType::U32 => "uint32_t(1)",
            VarType::I64 => "int64_t(1)",
            VarType::U64 => "uint64_t(1)",
            VarType::F32 => "float32_t(1)",
            VarType::F64 => "float64_t(1)",
            _ => todo!(),
        },
        ReduceOp::And => match ty {
            VarType::Bool => "uint8_t(1)",
            VarType::U8 => "uint8_t(0xff)",
            VarType::U16 => "uint16_t(0xffff)",
            VarType::U32 => "uint32_t(0xffffffff)",
            VarType::U64 => "uint64_t(0xfffffffffffffffful)",
            _ => todo!(),
        },
        ReduceOp::Or => match ty {
            VarType::Bool => "uint8_t(0)",
            VarType::U8 => "uint8_t(0x0)",
            VarType::U16 => "uint16_t(0x0)",
            VarType::U32 => "uint32_t(0x0)",
            VarType::U64 => "uint64_t(0x0)",
            _ => todo!(),
        },
        ReduceOp::Xor => match ty {
            VarType::Bool => "uint8_t(0)",
            VarType::U8 => "uint8_t(0x0)",
            VarType::U16 => "uint16_t(0x0)",
            VarType::U32 => "uint32_t(0x0)",
            VarType::U64 => "uint64_t(0x0)",
            _ => todo!(),
        },
        _ => todo!(),
    };

    let ty = glsl_ty(ty);

    let shader = device.get_shader_glsl(
        include_str!("kernels/reduce.glsl"),
        ShaderKind::Compute,
        &[
            ("REDUCE", Some(reduction)),
            ("INIT", Some(init)),
            ("TYPE", Some(ty)),
            ("WORK_GROUP_SIZE", Some("32")),
        ],
    );
    let pipeline = device.get_pipeline(&PipelineDesc {
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

    {
        let src = src.clone();
        let in_buffer = in_buffer.clone();
        rgraph
            .pass("Reduce copy to input buffer")
            .read(&src, AccessType::ComputeShaderReadOther)
            .write(&in_buffer, AccessType::ComputeShaderWrite)
            .record(move |device, cb, _| {
                unsafe {
                    device.cmd_copy_buffer(
                        cb,
                        src.vk(),
                        in_buffer.vk(),
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: src.info().size as _,
                        }],
                    )
                };
            });
    }

    log::trace!("Reducing {num} elements with {reduction}");
    log::trace!("Launching {n_passes} shader passes");
    log::trace!("Scratch Buffer size: {scratch_size}");
    for i in (0..n_passes).rev() {
        log::trace!("Launching shader");
        {
            let in_buffer = in_buffer.clone();
            let out_buffer = out_buffer.clone();
            let size_buffer = size_buffer.clone();
            let pipeline = pipeline.clone();
            rgraph
                .pass("Reduce: reduction")
                .read(&in_buffer, AccessType::ComputeShaderReadOther)
                .read(&size_buffer, AccessType::ComputeShaderReadOther)
                .write(&out_buffer, AccessType::ComputeShaderWrite)
                .record(move |device, cb, pool| {
                    pipeline.submit(
                        cb,
                        pool,
                        device,
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
                });
        }

        // Swap In/Out buffers
        std::mem::swap(&mut in_buffer, &mut out_buffer);
    }

    {
        let in_buffer = in_buffer.clone();
        let dst = dst.clone();
        rgraph
            .pass("Reduce: copy to output")
            .read(&in_buffer, AccessType::TransferRead)
            .write(&dst, AccessType::TransferWrite)
            .record(move |device, cb, _| {
                unsafe {
                    device.cmd_copy_buffer(
                        cb,
                        in_buffer.vk(),
                        dst.vk(),
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: ty_size as _,
                        }],
                    )
                };
            });
    }
}
