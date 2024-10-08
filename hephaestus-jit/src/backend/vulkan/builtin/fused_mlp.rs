use std::hash::Hash;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use vk_sync::AccessType;

use crate::backend::vulkan::builtin::utils::GlslShaderDef;
// use crate::backend::vulkan::codegen::CodegenDef;
use crate::backend::vulkan::vulkan_core::pipeline::{Pipeline, PipelineDef, ShaderKind};
use crate::backend::vulkan::{codegen, VulkanDevice};
use crate::utils::usize;
use crate::{
    backend::vulkan::vulkan_core::{
        buffer::{Buffer, BufferInfo},
        graph::RGraph,
    },
    vartype::FusedMlpConfig,
};

use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineInfo, WriteSet,
};

#[derive(Hash)]
enum Activation {
    ReLU,
}

#[allow(non_snake_case)]
#[derive(Hash)]
pub struct MLPCompileDef {
    width: u32,
    n_iters: u32,
    activation: Activation,
    inference: bool,

    in_width: u32,
    out_width: u32,

    input_layout: u32,
    output_layout: u32,

    hidden_mat: u32,
}
impl PipelineDef for MLPCompileDef {
    fn generate(self) -> PipelineInfo {
        let MLPCompileDef {
            width,
            n_iters,
            activation,
            inference,
            in_width,
            out_width,
            input_layout,
            output_layout,
            hidden_mat,
        } = self;

        let activation = match activation {
            Activation::ReLU => "RELU",
        };
        let code = GlslShaderDef {
            code: include_str!("kernels/fused_mlp_inference.glsl"),
            kind: ShaderKind::Compute,
            defines: &[
                ("WIDTH", Some(&format!("{width}"))),
                ("N_ITERS", Some(&format!("{n_iters}"))),
                ("INFERENCE", Some(&format!("{inference}"))),
                ("OUT_T", Some("float16_t")),
                ("ACTIVATION", Some(activation)),
                ("A_BITS", Some("16")),
                ("IN_WIDTH", Some(&format!("{in_width}"))),
                ("OUT_WIDTH", Some(&format!("{out_width}"))),
                ("INPUT_LAYOUT", Some(&format!("{input_layout}"))),
                ("OUTPUT_LAYOUT", Some(&format!("{output_layout}"))),
                ("HIDDEN_MAT", Some(&format!("{hidden_mat}"))),
            ],
        }
        .compile();
        let desc_set_layouts = [DescSetLayout {
            bindings: (0..=4)
                .map(|i| Binding {
                    binding: i,
                    count: 1,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                })
                .collect::<Vec<_>>(),
        }];
        PipelineInfo {
            code,
            desc_set_layouts: desc_set_layouts.into(),
        }
    }

    fn typed_hash(&self, state: &mut impl std::hash::Hasher) {
        std::any::TypeId::of::<Self>().hash(state);
        self.hash(state);
    }
}
// impl codegen::CodegenDef for MLPCompileDef {
//     fn generate(&self) -> Vec<u32> {
//         let MLPCompileDef {
//             width,
//             n_iters,
//             activation,
//             inference,
//             in_width,
//             out_width,
//             input_layout,
//             output_layout,
//             hidden_mat,
//         } = self;
//
//         let activation = match activation {
//             Activation::ReLU => "RELU",
//         };
//         GlslShaderDef {
//             code: include_str!("kernels/fused_mlp_inference.glsl"),
//             kind: crate::backend::vulkan::shader_cache::ShaderKind::Compute,
//             defines: &[
//                 ("WIDTH", Some(&format!("{width}"))),
//                 ("N_ITERS", Some(&format!("{n_iters}"))),
//                 ("INFERENCE", Some(&format!("{inference}"))),
//                 ("OUT_T", Some("float16_t")),
//                 ("ACTIVATION", Some(activation)),
//                 ("A_BITS", Some("16")),
//                 ("IN_WIDTH", Some(&format!("{in_width}"))),
//                 ("OUT_WIDTH", Some(&format!("{out_width}"))),
//                 ("INPUT_LAYOUT", Some(&format!("{input_layout}"))),
//                 ("OUTPUT_LAYOUT", Some(&format!("{output_layout}"))),
//                 ("HIDDEN_MAT", Some(&format!("{hidden_mat}"))),
//             ],
//         }
//         .generate()
//     }
// }

pub fn mlp_inference(
    device: &VulkanDevice,
    rgraph: &mut RGraph,

    input: Arc<Buffer>,
    weights: Arc<Buffer>,
    output: Arc<Buffer>,
    config: Option<Arc<Buffer>>,
    default_config: FusedMlpConfig,

    width: usize,
    in_width: usize,
    out_width: usize,

    hidden_layers: usize,
) {
    let n_iters = if width >= 256 { 2 } else { 8 };
    let n_block_rows = width / 16;

    // We don't support dynamic config yet
    assert!(config.is_none());
    let batch_size = default_config.batch_size as usize;

    assert!(
        batch_size % (16 * n_iters) == 0,
        "Batch size must be a multiple of {}.",
        16 * n_iters
    );

    // let threads = (32u32, n_block_rows as u32, 1u32); // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

    let n_elems_per_block = 16 * n_iters;
    let n_blocks = usize::div_round_up(batch_size, n_elems_per_block);

    let blocks = (n_blocks as u32, 1, 1);

    let pipeline = Pipeline::create(
        &device,
        MLPCompileDef {
            width: width as _,
            n_iters: n_iters as _,
            activation: Activation::ReLU,
            inference: true,
            in_width: in_width as _,
            out_width: out_width as _,
            input_layout: 0,
            output_layout: 0,
            hidden_mat: (hidden_layers - 1) as _,
        },
    );
    // let code = device.get_shader(&MLPCompileDef {
    //     width: width as _,
    //     n_iters: n_iters as _,
    //     activation: Activation::ReLU,
    //     inference: true,
    //     in_width: in_width as _,
    //     out_width: out_width as _,
    //     input_layout: 0,
    //     output_layout: 0,
    //     hidden_mat: (hidden_layers - 1) as _,
    // });
    //
    // let pipeline = device.get_pipeline(&PipelineInfo {
    //     code: &code,
    //     desc_set_layouts: &[DescSetLayout {
    //         bindings: &(0..=4)
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
                size: std::mem::size_of::<FusedMlpConfig>(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
        );
        config_buffer
            .mapped_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&[default_config]));

        Arc::new(config_buffer)
    });

    rgraph
        .pass("Fused MLP")
        .read(&config_buffer, AccessType::ComputeShaderReadOther)
        .read(&input, AccessType::ComputeShaderReadOther)
        .read(&weights, AccessType::ComputeShaderReadOther)
        .read(&output, AccessType::ComputeShaderReadOther)
        .write(&output, AccessType::ComputeShaderWrite)
        .record(move |device, cb, pool| {
            pipeline.submit(
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
                        buffers: &[BufferWriteInfo { buffer: &weights }],
                    },
                    WriteSet {
                        set: 0,
                        binding: 2,
                        buffers: &[BufferWriteInfo { buffer: &output }],
                    },
                    WriteSet {
                        set: 0,
                        binding: 3,
                        buffers: &[BufferWriteInfo {
                            buffer: &config_buffer,
                        }],
                    },
                ],
                blocks,
            );
        });
}
