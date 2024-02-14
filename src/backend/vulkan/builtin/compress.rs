use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use super::super::vulkan_core::{
    buffer::{Buffer, BufferInfo},
    graph::RGraph,
};
use super::prefix_sum::prefix_sum_scratch_buffer;
use super::utils::*;
use crate::backend::vulkan::shader_cache::ShaderKind;
use crate::backend::vulkan::VulkanDevice;
use crate::backend::vulkan::{
    pipeline::{Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet},
    vkdevice::LaunchConfig,
};
use crate::utils;

pub fn compress(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    num: usize,
    size_buffer: Option<Arc<Buffer>>,
    out_count: &Arc<Buffer>,
    src: &Arc<Buffer>,
    index_out: &Arc<Buffer>,
) {
    compress_large(
        device,
        rgraph,
        num as _,
        size_buffer,
        out_count,
        src,
        index_out,
    )
}

pub fn compress_small(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    num: u32,
    out_count: &Arc<Buffer>,
    src: &Arc<Buffer>,
    dst: &Arc<Buffer>,
) {
    const ITEMS_PER_THREAD: u32 = 4;
    let thread_count = utils::u32::round_pow2((num + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);

    let shader = device.get_shader_glsl(
        include_str!("kernels/compress_small.glsl"),
        ShaderKind::Compute,
        &[("WORK_GROUP_SIZE", Some(&format!("{thread_count}")))],
    );

    // TODO: in the end we might get the size buffer as an argument when suporting dynamic
    // indices
    let mut size_buffer = Buffer::create(
        device,
        BufferInfo {
            size: std::mem::size_of::<u32>(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
            ..Default::default()
        },
    );
    size_buffer
        .mapped_slice_mut()
        .copy_from_slice(bytemuck::cast_slice(&[num as u32]));
    let size_buffer = Arc::new(size_buffer);

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
                Binding {
                    binding: 3,
                    count: 1,
                },
            ],
        }],
    });

    log::trace!("Counting {num} elements with count_small");
    {
        let src = src.clone();
        let size_buffer = size_buffer.clone();
        let dst = dst.clone();
        let out_count = out_count.clone();
        rgraph
            .pass("Compress Small")
            .read(&src, AccessType::ComputeShaderReadOther)
            .read(&size_buffer, AccessType::ComputeShaderReadOther)
            .write(&dst, AccessType::ComputeShaderWrite)
            .write(&out_count, AccessType::ComputeShaderWrite)
            .record(move |device, cb, pool| {
                pipeline.submit(
                    cb,
                    pool,
                    device,
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
                            buffers: &[BufferWriteInfo { buffer: &out_count }],
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
            });
    }
}
pub fn compress_large(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    num: usize,
    size_buffer: Option<Arc<Buffer>>,
    out_count: &Arc<Buffer>,
    src: &Arc<Buffer>,
    dst: &Arc<Buffer>,
) {
    let items_per_thread = 16;
    let block_size = 128;
    let items_per_block = items_per_thread * block_size;
    let block_count = (num + items_per_block - 1) / items_per_block;
    let warp_size = device
        .device
        .physical_device
        .subgroup_properties
        .subgroup_size as usize;

    let scratch_items = 1 + warp_size + block_count;

    let compress_large = device.get_shader_glsl(
        include_str!("kernels/compress_large.glsl"),
        ShaderKind::Compute,
        &[
            ("WORK_GROUP_SIZE", Some(&format!("{block_size}"))),
            ("INIT", Some("")),
        ],
    );
    let compress_large = device.get_pipeline(&PipelineDesc {
        code: &compress_large,
        desc_set_layouts: &[DescSetLayout {
            bindings: &(0..5)
                .map(|i| Binding {
                    binding: i,
                    count: 1,
                })
                .collect::<Vec<_>>(),
        }],
    });

    let size_buffer = size_buffer.unwrap_or_else(|| {
        let mut size_buffer = Buffer::create(
            device,
            BufferInfo {
                size: std::mem::size_of::<u32>(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
        );

        size_buffer
            .mapped_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&[num as u32]));
        Arc::new(size_buffer)
    });

    let scratch_buffer = prefix_sum_scratch_buffer(device, rgraph, scratch_items as _);

    {
        let size_buffer = size_buffer.clone();
        let src = src.clone();
        let scratch_buffer = scratch_buffer.clone();
        let dst = dst.clone();
        let out_count = out_count.clone();
        rgraph
            .pass("Compress Large")
            .read(&size_buffer, AccessType::ComputeShaderReadOther)
            .read(&src, AccessType::ComputeShaderReadOther)
            .read(&scratch_buffer, AccessType::ComputeShaderReadOther)
            .write(&out_count, AccessType::ComputeShaderWrite)
            .write(&scratch_buffer, AccessType::ComputeShaderWrite)
            .write(&dst, AccessType::ComputeShaderWrite)
            .record(move |device, cb, pool| {
                compress_large.submit(
                    cb,
                    pool,
                    device,
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
                            buffers: &[BufferWriteInfo { buffer: &out_count }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 3,
                            buffers: &[BufferWriteInfo {
                                buffer: &size_buffer,
                            }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 4,
                            buffers: &[BufferWriteInfo {
                                buffer: &scratch_buffer,
                            }],
                        },
                    ],
                    (block_count as _, 1, 1),
                );
            });
    }
}
