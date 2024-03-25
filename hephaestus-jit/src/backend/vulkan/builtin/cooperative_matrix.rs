use std::hash::Hash;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use crate::backend::vulkan::builtin::utils::{component_type, GlslShaderDef};
use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineInfo, WriteSet,
};
use crate::backend::vulkan::vulkan_core::pipeline::{Pipeline, PipelineDef, ShaderKind};
use crate::backend::vulkan::{codegen, VulkanDevice};
use crate::vartype::VarType;
use crate::{
    backend::vulkan::vulkan_core::{
        buffer::{Buffer, BufferInfo},
        graph::{RGraph, ResourceId},
    },
    vartype::MatMulConfig,
};

use super::utils::glsl_ty;

#[allow(non_snake_case)]
#[derive(Hash)]
struct CoopMMADef<'a> {
    lM: u32,
    lN: u32,
    lK: u32,
    TILE_M: u32,
    TILE_N: u32,
    TILE_K: u32,
    a_bits: usize,
    a_type: &'a str,
    c_bits: usize,
    c_type: &'a str,
    subgroup_size: u32,
}
impl<'a> PipelineDef for CoopMMADef<'a> {
    fn generate(self) -> PipelineInfo {
        let CoopMMADef {
            lM,
            lN,
            lK,
            TILE_M,
            TILE_N,
            TILE_K,
            a_bits,
            a_type,
            c_bits,
            c_type,
            subgroup_size,
        } = self;
        let code = GlslShaderDef {
            code: include_str!("kernels/cooperative_matrix_sh.glsl"),
            kind: ShaderKind::Compute,
            defines: &[
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
        }
        .compile();

        let layout = [DescSetLayout {
            bindings: (0..5)
                .map(|i| Binding {
                    binding: i,
                    count: 1,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                })
                .collect::<Vec<_>>(),
        }];
        PipelineInfo {
            code,
            desc_set_layouts: layout.into(),
        }
    }

    fn typed_hash(&self, state: &mut impl std::hash::Hasher) {
        std::any::TypeId::of::<CoopMMADef<'static>>().hash(state);
        Hash::hash(self, state);
    }
}

#[allow(non_snake_case)]
pub fn multiply(
    device: &VulkanDevice,
    rgraph: &mut RGraph,
    a_type: &'static VarType,
    c_type: &'static VarType,
    M: u32,
    N: u32,
    K: u32,
    config: Option<ResourceId>,
    mat_a: ResourceId,
    mat_b: ResourceId,
    mat_c: ResourceId,
    mat_d: ResourceId,
) {
    let subgroup_size = device.physical_device.subgroup_properties.subgroup_size;

    let coopmat_type = device
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

    // TODO: Dynamically get TILE size
    let TILE_M = 256;
    let TILE_N = 256;
    let TILE_K = 32;

    let a_bits = a_type.size() * 8;
    let c_bits = c_type.size() * 8;

    let a_type = glsl_ty(a_type);
    let c_type = glsl_ty(c_type);

    let dispatch_x = N / TILE_N;
    let dispatch_y = M / TILE_M;

    log::trace!("Cooperative Matrix Multiply Add: ");
    log::trace!("M={M}, N={N}, K={K}, lM={lM}, lN={lN}, lK={lK}, TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}");
    log::trace!("Using cooperative matrix type: {coopmat_type:#?}");
    log::trace!("Dispatch: ( {dispatch_x}, {dispatch_y}, 1 )");

    let pipeline = Pipeline::create(
        &device,
        CoopMMADef {
            lM,
            lN,
            lK,
            TILE_M,
            TILE_N,
            TILE_K,
            a_bits,
            a_type,
            c_bits,
            c_type,
            subgroup_size,
        },
    );

    // let code = device.get_shader(&CoopMMADef {
    //     lM,
    //     lN,
    //     lK,
    //     TILE_M,
    //     TILE_N,
    //     TILE_K,
    //     a_bits,
    //     a_type,
    //     c_bits,
    //     c_type,
    //     subgroup_size,
    // });

    // let pipeline = device.get_pipeline(&PipelineInfo {
    //     code: &code,
    //     desc_set_layouts: &[DescSetLayout {
    //         bindings: &(0..5)
    //             .map(|i| Binding {
    //                 binding: i,
    //                 count: 1,
    //                 ty: vk::DescriptorType::STORAGE_BUFFER,
    //             })
    //             .collect::<Vec<_>>(),
    //     }],
    // });

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
            .copy_from_slice(bytemuck::cast_slice(&[MatMulConfig { M, N, K }]));

        rgraph.external(&Arc::new(config_buffer))
    });

    {
        rgraph
            .pass("Cooperative Matrix Multiply")
            .read(config_buffer, AccessType::ComputeShaderReadOther)
            .read(mat_a, AccessType::ComputeShaderReadOther)
            .read(mat_b, AccessType::ComputeShaderReadOther)
            .read(mat_c, AccessType::ComputeShaderReadOther)
            .read(mat_d, AccessType::ComputeShaderReadOther)
            .write(mat_d, AccessType::ComputeShaderWrite)
            .record(move |device, cb, ctx| {
                pipeline.submit(
                    cb,
                    ctx,
                    device,
                    &[
                        WriteSet {
                            set: 0,
                            binding: 0,
                            buffers: &[BufferWriteInfo {
                                buffer: ctx.buffer(config_buffer),
                            }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 1,
                            buffers: &[BufferWriteInfo {
                                buffer: ctx.buffer(mat_a),
                            }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 2,
                            buffers: &[BufferWriteInfo {
                                buffer: ctx.buffer(mat_b),
                            }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 3,
                            buffers: &[BufferWriteInfo {
                                buffer: ctx.buffer(mat_c),
                            }],
                        },
                        WriteSet {
                            set: 0,
                            binding: 4,
                            buffers: &[BufferWriteInfo {
                                buffer: &ctx.buffer(mat_d),
                            }],
                        },
                    ],
                    (dispatch_x, dispatch_y, 1),
                );
            });
    }
}
