use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};
use crate::backend::vulkan::shader_cache::ShaderKind;
use crate::backend::vulkan::vkdevice::{glsl_short_ty, glsl_ty};
use crate::backend::vulkan::vulkan_core::buffer::BufferInfo;
use crate::op::ReduceOp;
use crate::vartype::VarType;

use super::pool::{Lease, Pool};
use super::vkdevice::LaunchConfig;
use super::vulkan_core::buffer::Buffer;
use super::VulkanDevice;

impl VulkanDevice {
    pub fn prefix_sum(
        &self,
        cb: vk::CommandBuffer,
        pool: &mut Pool,
        ty: &VarType,
        num: usize,
        inclusive: bool,
        src: &Buffer,
        dst: &Buffer,
    ) {
        self.prefix_sum_large(cb, pool, ty, num, inclusive, src, dst);
    }

    pub fn prefix_sum_large(
        &self,
        cb: vk::CommandBuffer,
        pool: &mut Pool,
        ty: &VarType,
        num: usize,
        inclusive: bool,
        input: &Buffer,
        output: &Buffer,
    ) {
        // let elem_size = std::mem::size_of::<u32>();
        let num = num;

        let vector_size = 4; // M
        let loads_per_thread = 4; // N
        let items_per_thread = loads_per_thread * vector_size;
        let block_size = 128;
        let warp_size = self
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

        let prefix_sum_large = self.get_shader_glsl(
            include_str!("kernels/prefix_sum_large.glsl"),
            ShaderKind::Compute,
            &[
                ("WORK_GROUP_SIZE", Some(&format!("{block_size}"))),
                ("T", Some(&glsl_ty)),
                ("VT", Some(&vec_ty)),
                ("M", Some(&format!("{vector_size}"))),
                ("N", Some(&format!("{loads_per_thread}"))),
                ("INCLUSIVE", inclusive.then(|| "")),
                ("INIT", Some("")),
            ],
        );

        let pipeline = self.get_pipeline(&PipelineDesc {
            code: &prefix_sum_large,
            desc_set_layouts: &[DescSetLayout {
                bindings: &(0..4)
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

        // NOTE: don't need barrier here, as scratch buffer init is not depending on anything

        let scratch_buffer = self.prefix_sum_scratch_buffer(cb, pool, scratch_items);

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

        pipeline.submit(
            cb,
            pool,
            self,
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
    }

    pub fn prefix_sum_scratch_buffer(
        &self,
        cb: vk::CommandBuffer,
        pool: &mut Pool,
        scratch_items: usize,
    ) -> Lease<Buffer> {
        let LaunchConfig {
            block_size,
            grid_size,
        } = self.get_launch_config(scratch_items);

        let prefix_sum_large_init = self.get_shader_glsl(
            include_str!("kernels/prefix_sum_large_init.glsl"),
            ShaderKind::Compute,
            &[("WORK_GROUP_SIZE", Some(&format!("{block_size}")))],
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

        let scratch_buffer = pool.lease_buffer(BufferInfo {
            size: std::mem::size_of::<u64>() * scratch_items as usize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
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
            .copy_from_slice(bytemuck::cast_slice(&[scratch_items as u32]));

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
                        buffer: &size_buffer,
                    }],
                },
            ],
            (grid_size, 1, 1),
        );

        scratch_buffer
    }
}
