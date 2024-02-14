use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};
use crate::backend::vulkan::shader_cache::ShaderKind;
use crate::backend::vulkan::VulkanDevice;
use crate::vartype::VarType;
use crate::{
    backend::vulkan::vulkan_core::{
        buffer::{Buffer, BufferInfo},
        graph::RGraph,
    },
    vartype::MatMulConfig,
};

use super::utils::glsl_ty;

pub fn multiply(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    ty: &'static VarType,
    max_n: u32,
    max_m: u32,
    max_k: u32,
    config: Option<Arc<Buffer>>,
    mat_a: Arc<Buffer>,
    mat_b: Arc<Buffer>,
    mat_c: Arc<Buffer>,
) {
    log::trace!("Cooperative Matrix");
    let subgroup_size = device
        .device
        .physical_device
        .subgroup_properties
        .subgroup_size;

    let subgroups_per_workgroup = 1;

    let workgroup_size = subgroup_size * subgroups_per_workgroup;

    let coop_n = 16;
    let coop_m = 16;
    let coop_k = 16;

    // let num_tiles = max_n * max_m;

    let tiles_n = max_n / coop_n;
    let tiles_m = max_m / coop_m;
    let num_tiles = tiles_n * tiles_m;

    let num_workgroups = num_tiles / subgroups_per_workgroup;

    log::trace!("Workgroups: {num_workgroups}");

    let glsl_ty = glsl_ty(ty);
    let code = device.get_shader_glsl(
        include_str!("kernels/cooperative_matrix.glsl"),
        ShaderKind::Compute,
        &[
            ("WORKGROUP_SIZE", Some(&format!("{workgroup_size}"))),
            ("T", Some(&glsl_ty)),
            ("COOP_N", Some(&format!("{coop_n}"))),
            ("COOP_M", Some(&format!("{coop_m}"))),
            ("COOP_K", Some(&format!("{coop_k}"))),
        ],
    );
    let pipeline = device.get_pipeline(&PipelineDesc {
        code: &code,
        desc_set_layouts: &[DescSetLayout {
            bindings: &(0..4)
                .map(|i| Binding {
                    binding: i,
                    count: 1,
                })
                .collect::<Vec<_>>(),
        }],
    });

    let config_buffer = config.unwrap_or_else(|| {
        let mut config_buffer = Buffer::create(
            device,
            BufferInfo {
                size: std::mem::size_of::<MatMulConfig>(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
        );
        config_buffer
            .mapped_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&[MatMulConfig {
                N: max_n,
                M: max_m,
                K: max_k,
            }]));

        Arc::new(config_buffer)
    });

    {
        rgraph
            .pass("Cooperative Matrix Multiply")
            .read(&config_buffer, AccessType::ComputeShaderReadOther)
            .read(&mat_a, AccessType::ComputeShaderReadOther)
            .read(&mat_b, AccessType::ComputeShaderReadOther)
            .read(&mat_c, AccessType::ComputeShaderReadOther)
            .write(&mat_c, AccessType::ComputeShaderWrite)
            .record(move |device, cb, pool| {
                pipeline.submit(
                    cb,
                    pool,
                    device,
                    &[
                        WriteSet {
                            set: 0,
                            binding: 0,
                            buffers: &[BufferWriteInfo {
                                buffer: &config_buffer,
                            }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 1,
                            buffers: &[BufferWriteInfo { buffer: &mat_a }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 2,
                            buffers: &[BufferWriteInfo { buffer: &mat_b }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 3,
                            buffers: &[BufferWriteInfo { buffer: &mat_c }],
                        },
                    ],
                    (num_workgroups, 1, 1),
                );
            });
    }
}
