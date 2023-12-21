use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::backend::vulkan::buffer::BufferInfo;
use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};
use crate::backend::vulkan::shader_cache::ShaderKind;
use crate::backend::vulkan::vkdevice::round_pow2;

use super::buffer::Buffer;
use super::pool::Pool;
use super::VulkanDevice;

impl VulkanDevice {
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
        num: usize,
        out_count: &Buffer,
        src: &Buffer,
        dst: &Buffer,
    ) {
        let items_per_thread = 16;
        let block_size = 128;
        let items_per_block = items_per_thread * block_size;
        let block_count = (num + items_per_block - 1) / items_per_block;
        let warp_size = self
            .device
            .physical_device
            .subgroup_properties
            .subgroup_size as usize;

        let scratch_items = 1 + warp_size + block_count;
        let trailer = items_per_block * block_count - num;
        dbg!(block_count);

        let compress_large = self.get_shader_glsl(
            include_str!("kernels/compress_large.glsl"),
            ShaderKind::Compute,
            &[
                ("WORK_GROUP_SIZE", Some(&format!("{block_size}"))),
                ("INIT", Some("")),
            ],
        );
        let compress_large = self.get_pipeline(&PipelineDesc {
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

        let scratch_buffer = self.prefix_sum_scratch_buffer(cb, pool, scratch_items as _);

        // Barrier
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
    }
}
