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
    pipeline::{Binding, BufferWriteInfo, DescSetLayout, PipelineInfo, WriteSet},
    vkdevice::LaunchConfig,
};
use crate::backend::vulkan::{shader_cache::ShaderKind, vulkan_core::pipeline::Pipeline};
use crate::vartype::VarType;

pub fn prefix_sum(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    ty: &VarType,
    num: usize,
    inclusive: bool,
    input: &Arc<Buffer>,
    output: &Arc<Buffer>,
) {
    prefix_sum_large(device, rgraph, ty, num, inclusive, input, output)
}
pub fn prefix_sum_large(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    ty: &VarType,
    num: usize,
    inclusive: bool,
    input: &Arc<Buffer>,
    output: &Arc<Buffer>,
) {
    let vector_size = 4; // M
    let loads_per_thread = 4; // N
    let items_per_thread = loads_per_thread * vector_size;
    let block_size = 128;
    let warp_size = device
        .device
        .physical_device
        .subgroup_properties
        .subgroup_size as usize;
    let items_per_block = items_per_thread * block_size;
    let block_count = (num + items_per_block - 1) / items_per_block;
    let scratch_items = 1 + warp_size + block_count;

    let glsl_ty = glsl_ty(ty);
    let glsl_short_ty = glsl_short_ty(ty);
    let vec_ty = format!("{glsl_short_ty}vec{vector_size}");

    let prefix_sum_large = Pipeline::create(
        &device,
        &GlslShaderDef {
            code: &include_str!("kernels/prefix_sum_large.glsl"),
            kind: ShaderKind::Compute,
            defines: &[
                ("WORK_GROUP_SIZE", Some(&format!("{block_size}"))),
                ("T", Some(&glsl_ty)),
                ("VT", Some(&vec_ty)),
                ("M", Some(&format!("{vector_size}"))),
                ("N", Some(&format!("{loads_per_thread}"))),
                ("INCLUSIVE", inclusive.then(|| "")),
                ("INIT", Some("")),
            ],
        },
    );
    // let prefix_sum_large = device.get_shader_glsl(
    //     include_str!("kernels/prefix_sum_large.glsl"),
    //     ShaderKind::Compute,
    //     &[
    //         ("WORK_GROUP_SIZE", Some(&format!("{block_size}"))),
    //         ("T", Some(&glsl_ty)),
    //         ("VT", Some(&vec_ty)),
    //         ("M", Some(&format!("{vector_size}"))),
    //         ("N", Some(&format!("{loads_per_thread}"))),
    //         ("INCLUSIVE", inclusive.then(|| "")),
    //         ("INIT", Some("")),
    //     ],
    // );
    //
    // let pipeline = device.get_pipeline(&PipelineInfo {
    //     code: &prefix_sum_large,
    //     desc_set_layouts: &[DescSetLayout {
    //         bindings: &(0..4)
    //             .map(|i| Binding {
    //                 binding: i,
    //                 count: 1,
    //                 ty: vk::DescriptorType::STORAGE_BUFFER,
    //             })
    //             .collect::<Vec<_>>(),
    //     }],
    // });

    let mut size_buffer = Buffer::create(
        &device,
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

    // NOTE: don't need barrier here, as scratch buffer init is not depending on anything

    let scratch_buffer = prefix_sum_scratch_buffer(device, rgraph, scratch_items);

    {
        let output = output.clone();
        let input = input.clone();
        rgraph
            .pass("Prefix Sum Large")
            .read(&size_buffer, AccessType::ComputeShaderReadOther)
            .read(&input, AccessType::ComputeShaderReadOther)
            .read(&scratch_buffer, AccessType::ComputeShaderReadOther)
            .write(&scratch_buffer, AccessType::ComputeShaderWrite)
            .write(&output, AccessType::ComputeShaderWrite)
            .record(move |device, cb, pool| {
                prefix_sum_large.submit(
                    cb,
                    pool,
                    device,
                    &[
                        WriteSet {
                            set: 0,
                            binding: 0,
                            buffers: &[BufferWriteInfo { buffer: &input }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 1,
                            buffers: &[BufferWriteInfo { buffer: &output }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 2,
                            buffers: &[BufferWriteInfo {
                                buffer: &size_buffer,
                            }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 3,
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

pub fn prefix_sum_scratch_buffer(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    scratch_items: usize,
) -> Arc<Buffer> {
    let LaunchConfig {
        block_size,
        grid_size,
    } = device.get_launch_config(scratch_items);

    let prefix_sum_large_init = Pipeline::create(
        &device,
        &GlslShaderDef {
            code: &include_str!("kernels/prefix_sum_large_init.glsl"),
            kind: ShaderKind::Compute,
            defines: &[("WORK_GROUP_SIZE", Some(&format!("{block_size}")))],
        },
    );

    // let prefix_sum_large_init = device.get_shader_glsl(
    //     include_str!("kernels/prefix_sum_large_init.glsl"),
    //     ShaderKind::Compute,
    //     &[("WORK_GROUP_SIZE", Some(&format!("{block_size}")))],
    // );
    // let prefix_sum_large_init = device.get_pipeline(&PipelineInfo {
    //     code: &prefix_sum_large_init,
    //     desc_set_layouts: &[DescSetLayout {
    //         bindings: &[
    //             Binding {
    //                 binding: 0,
    //                 count: 1,
    //                 ty: vk::DescriptorType::STORAGE_BUFFER,
    //             },
    //             Binding {
    //                 binding: 1,
    //                 count: 1,
    //                 ty: vk::DescriptorType::STORAGE_BUFFER,
    //             },
    //         ],
    //     }],
    // });

    let scratch_buffer = Arc::new(Buffer::create(
        device,
        BufferInfo {
            size: std::mem::size_of::<u64>() * scratch_items as usize,
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
        .copy_from_slice(bytemuck::cast_slice(&[scratch_items as u32]));
    let size_buffer = Arc::new(size_buffer);

    {
        let scratch_buffer = scratch_buffer.clone();
        rgraph
            .pass("Initialize prefix sum scratch buffer")
            .read(&size_buffer, AccessType::ComputeShaderReadOther)
            .write(&scratch_buffer, AccessType::ComputeShaderWrite)
            .record(move |device, cb, pool| {
                prefix_sum_large_init.submit(
                    cb,
                    pool,
                    device,
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
                                buffer: &size_buffer,
                            }],
                        },
                    ],
                    (grid_size, 1, 1),
                );
            });
    }

    scratch_buffer
}
