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
    pub fn compress_small(
        &self,
        cb: vk::CommandBuffer,
        pool: &mut Pool,
        num: u32,
        out_count: &Buffer,
        src: &Buffer,
        dst: &Buffer,
    ) {
        const ITEMS_PER_THREAD: u32 = 4;
        let thread_count = round_pow2((num + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
        dbg!(thread_count);

        let shader = self.get_shader_glsl(
            include_str!("kernels/compress_small.glsl"),
            ShaderKind::Compute,
            &[("WORK_GROUP_SIZE", Some(&format!("{thread_count}")))],
        );

        // TODO: in the end we might get the size buffer as an argument when suporting dynamic
        // indices
        let mut size_buffer = pool.buffer(BufferInfo {
            size: std::mem::size_of::<u32>(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
            ..Default::default()
        });
        size_buffer
            .mapped_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&[num as u32]));

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
                    Binding {
                        binding: 3,
                        count: 1,
                    },
                ],
            }],
        });

        let memory_barriers = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build()];
        unsafe {
            self.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &memory_barriers,
                &[],
                &[],
            );
        }
        log::trace!("Counting {num} elements with count_small");

        pipeline.submit(
            cb,
            pool,
            self,
            &[
                WriteSet {
                    set: 0,
                    binding: 0,
                    buffers: &[BufferWriteInfo { buffer: src }],
                },
                WriteSet {
                    set: 0,
                    binding: 1,
                    buffers: &[BufferWriteInfo { buffer: dst }],
                },
                WriteSet {
                    set: 0,
                    binding: 2,
                    buffers: &[BufferWriteInfo { buffer: out_count }],
                },
                WriteSet {
                    set: 0,
                    binding: 3,
                    buffers: &[BufferWriteInfo {
                        buffer: &size_buffer,
                    }],
                },
            ],
            (1, 1, 1),
        );
    }
    pub fn compress_large(
        &self,
        cb: vk::CommandBuffer,
        pool: &mut Pool,
        num: u32,
        out_count: &Buffer,
        src: &Buffer,
        dst: &Buffer,
    ) {
        let items_per_thread = 16;
        let thread_count = 128;
        let items_per_block = items_per_thread * thread_count;
        let block_count = (num + items_per_block - 1) / items_per_block;
        let scratch_items = block_count + 32;
        let trailer = items_per_block * block_count - num;

        let compress_large = self.get_shader_glsl(
            include_str!("kernels/compress_large.glsl"),
            ShaderKind::Compute,
            &[("WORK_GROUP_SIZE", Some(&format!("{thread_count}")))],
        );
        let compress_large = self.get_pipeline(&PipelineDesc {
            code: &compress_large,
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
                    Binding {
                        binding: 3,
                        count: 1,
                    },
                    Binding {
                        binding: 4,
                        count: 1,
                    },
                ],
            }],
        });
        let prefix_sum_large_init = self.get_shader_glsl(
            include_str!("kernels/prefix_sum_large_init.glsl"),
            ShaderKind::Compute,
            &[("WORK_GROUP_SIZE", Some(&format!("{thread_count}")))],
        );
        let prefix_sum_large_init = self.get_pipeline(&PipelineDesc {
            code: &prefix_sum_large_init,
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
                ],
            }],
        });

        let scratch_buffer = pool.buffer(BufferInfo {
            size: std::mem::size_of::<u64>() * scratch_items as usize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        });

        let size_buffer = pool.buffer(BufferInfo {
            size: std::mem::size_of::<u64>() * scratch_items as usize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
            ..Default::default()
        });
        let scratch_items_size = pool.buffer(BufferInfo {
            size: std::mem::size_of::<u64>() * scratch_items as usize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
            ..Default::default()
        });

        // Initializing scratch buffer
        let memory_barriers = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build()];
        unsafe {
            self.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &memory_barriers,
                &[],
                &[],
            );
        }

        prefix_sum_large_init.submit(
            cb,
            pool,
            self,
            &[
                WriteSet {
                    set: 0,
                    binding: 0,
                    buffers: &[BufferWriteInfo {
                        buffer: &scratch_buffer,
                    }],
                },
                WriteSet {
                    set: 0,
                    binding: 1,
                    buffers: &[BufferWriteInfo {
                        buffer: &scratch_items_size,
                    }],
                },
            ],
            (scratch_items / thread_count, 1, 1),
        );

        // Compress
        let memory_barriers = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build()];
        unsafe {
            self.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &memory_barriers,
                &[],
                &[],
            );
        }

        compress_large.submit(
            cb,
            pool,
            self,
            &[
                WriteSet {
                    set: 0,
                    binding: 0,
                    buffers: &[BufferWriteInfo { buffer: &src }],
                },
                WriteSet {
                    set: 0,
                    binding: 1,
                    buffers: &[BufferWriteInfo { buffer: &dst }],
                },
                WriteSet {
                    set: 0,
                    binding: 2,
                    buffers: &[BufferWriteInfo {
                        buffer: &scratch_buffer,
                    }],
                },
                WriteSet {
                    set: 0,
                    binding: 3,
                    buffers: &[BufferWriteInfo { buffer: &out_count }],
                },
            ],
            (block_count, 1, 1),
        );
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
