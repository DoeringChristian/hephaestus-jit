use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use crate::backend::vulkan::builtin::utils::component_type;
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

#[allow(non_snake_case)]
pub fn multiply(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    a_type: &'static VarType,
    c_type: &'static VarType,
    M: u32,
    N: u32,
    K: u32,
    config: Option<Arc<Buffer>>,
    mat_a: Arc<Buffer>,
    mat_b: Arc<Buffer>,
    mat_c: Arc<Buffer>,
    mat_d: Arc<Buffer>,
) {
    let subgroup_size = device
        .device
        .physical_device
        .subgroup_properties
        .subgroup_size;

    let coopmat_type = device
        .device
        .cooperative_matrix_properties
        .iter()
        .find(|p| {
            p.a_type == p.b_type
                && p.c_type == p.result_type
                && component_type(a_type) == p.a_type
                && component_type(c_type) == p.c_type
                && p.scope == vk::ScopeKHR::SUBGROUP
        })
        .unwrap();

    let lM = coopmat_type.m_size;
    let lN = coopmat_type.n_size;
    let lK = coopmat_type.k_size;

    let COOP_PER_TILE_M = 8;
    let COOP_PER_TILE_N = 8;
    let COOP_PER_TILE_K = 2;

    let TILE_M = lM * COOP_PER_TILE_M;
    let TILE_N = lN * COOP_PER_TILE_N;
    let TILE_K = lK * COOP_PER_TILE_K;

    let a_bits = a_type.size() * 8;
    let c_bits = c_type.size() * 8;

    let a_type = glsl_ty(a_type);
    let c_type = glsl_ty(c_type);

    let dispatch_x = N / TILE_N;
    let dispatch_y = M / TILE_M;

    log::trace!("Cooperative Matrix Multiply Add: ");
    log::trace!("M={M}, N={N}, K={K}, lM={lM}, lN={lN}, lK={lK}, TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}");
    log::trace!("Dispatch: ( {dispatch_x}, {dispatch_y}, 1 )");

    let code = device.get_shader_glsl(
        include_str!("kernels/cooperative_matrix_sh.glsl"),
        ShaderKind::Compute,
        &[
            ("lM", Some(&format!("{lM}"))),
            ("lN", Some(&format!("{lN}"))),
            ("lK", Some(&format!("{lK}"))),
            ("TILE_M", Some(&format!("{TILE_M}"))),
            ("TILE_N", Some(&format!("{TILE_N}"))),
            ("TILE_K", Some(&format!("{TILE_K}"))),
            ("A_TYPE", Some(&format!("{a_type}"))),
            ("A_BITS", Some(&format!("{a_bits}"))),
            ("C_TYPE", Some(&format!("{c_type}"))),
            ("C_BITS", Some(&format!("{c_bits}"))),
            ("SUBGROUP_SIZE", Some(&format!("{subgroup_size}"))),
        ],
    );
    let pipeline = device.get_pipeline(&PipelineDesc {
        code: &code,
        desc_set_layouts: &[DescSetLayout {
            bindings: &(0..5)
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
            .copy_from_slice(bytemuck::cast_slice(&[MatMulConfig { N, M, K }]));

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
                        WriteSet {
                            set: 0,
                            binding: 4,
                            buffers: &[BufferWriteInfo { buffer: &mat_d }],
                        },
                    ],
                    (dispatch_x, dispatch_y, 1),
                );
            });
    }
}
