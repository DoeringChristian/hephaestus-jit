use std::sync::Arc;

use crate::backend::vulkan::builtin::utils::GlslShaderDef;
use crate::backend::vulkan::codegen::CodegenDef;
use crate::backend::vulkan::{codegen, VulkanDevice};
use crate::utils::usize;
use crate::{
    backend::vulkan::vulkan_core::{
        buffer::{Buffer, BufferInfo},
        graph::RGraph,
    },
    vartype::MatMulConfig,
};

use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
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
}
impl codegen::CodegenDef for MLPCompileDef {
    fn generate(&self) -> Vec<u32> {
        let MLPCompileDef {
            width,
            n_iters,
            activation,
            inference,
        } = self;

        let activation = match activation {
            Activation::ReLU => "RELU",
        };
        GlslShaderDef {
            code: include_str!("kernels/fused_mlp.glsl"),
            kind: crate::backend::vulkan::shader_cache::ShaderKind::Compute,
            defines: &[
                ("WIDTH", Some(&format!("{width}"))),
                ("N_ITERS", Some(&format!("{n_iters}"))),
                ("INFERENCE", Some(&format!("{inference}"))),
                ("OUT_T", Some("float16_t")),
                ("ACTIVATION", Some(activation)),
                ("A_BITS", Some("16")),
            ],
        }
        .generate()
    }
}

pub fn mlp_forward(
    device: &VulkanDevice,
    rgraph: &mut RGraph,

    input: Arc<Buffer>,
    weights: Arc<Buffer>,
    out_intermediary: Arc<Buffer>,
    output: Arc<Buffer>,
    config: Option<Arc<Buffer>>,

    batch_size: usize,
    in_width: usize,
    n_hidden_layers: usize,
    width: usize,
) {
    let n_iters = if width >= 256 { 2 } else { 8 };
    let n_block_rows = width / 16;

    assert!(
        batch_size % (16 * n_iters) == 0,
        "Batch size must be a multiple of {}.",
        16 * n_iters
    );

    let threads = (32u32, n_block_rows as u32, 1u32); // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

    let n_elems_per_block = 16 * n_iters;
    let n_blocks = usize::div_round_up(batch_size, n_elems_per_block);

    let blocks = (n_blocks, 1, 1);

    let code = device.get_shader(&MLPCompileDef {
        width: width as _,
        n_iters: n_iters as _,
        activation: Activation::ReLU,
        inference: true,
    });

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

    todo!()
}
