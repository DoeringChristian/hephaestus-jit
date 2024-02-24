#[cfg(test)]
use std::path::Path;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use half::f16;

use crate::backend::vulkan::vulkan_core::buffer::{Buffer, BufferInfo};
use crate::backend::vulkan::vulkan_core::graph::RGraph;
use crate::backend::vulkan::VulkanDevice;

use super::super::fused_mlp::*;
#[test]
fn fused_mlp_inference() {
    pretty_env_logger::try_init().ok();
    let device = VulkanDevice::create(0).unwrap();

    // [64] -> [64x64] -> ( _/ ) -> [64x64] -> [64]
    let n_inputs = 64;
    let n_outputs = 64;
    let hidden_layers = 0;
    let width = 64;
    let batch_size = 128;

    let path = Path::new(file!());
    let path = path.parent().unwrap().join("data");

    let input = std::fs::read(path.join("input.bin")).unwrap();
    assert_eq!(
        input.len(),
        (n_inputs * batch_size) * std::mem::size_of::<f16>()
    );
    let input = bytemuck::cast_slice::<_, f16>(&input).to_vec();

    let weights = std::fs::read(path.join("weights.bin")).unwrap();
    assert_eq!(
        weights.len(),
        width * width * (2 + hidden_layers as usize) * std::mem::size_of::<f16>()
    );
    let weights = bytemuck::cast_slice::<_, f16>(&weights).to_vec();

    let reference = std::fs::read(path.join("output.bin")).unwrap();
    let reference = bytemuck::cast_slice::<_, f16>(&reference).to_vec();
    // dbg!(bytemuck::cast_slice::<_, f16>(&input)[1]);
    // dbg!(bytemuck::cast_slice::<_, f16>(&weights)[1]);
    // dbg!(reference[1]);
    // let input = vec![f16::from_f32(1f32); n_inputs * batch_size];
    // let output = vec![f16::from_f32(1f32); 64];
    // let output_intermediate = vec![f16::from_f32(0.); 64 * (n_hidden_layers + 2)];
    // let mut win = vec![f16::from_f32(0f32); 64 * 64];

    // let mut wout = vec![f16::from_f32(0f32); 64 * 64];
    //
    // // // Initialize win and wout
    // for row in 0..64 {
    //     for col in 0..64 {
    //         if row == col {
    //             // win[64 * row + col] = f16::from_f32(1.);
    //             wout[64 * row + col] = f16::from_f32(1.);
    //         }
    //     }
    // }
    // wout[1] = f16::from_f32(1.0);
    // wout[0] = f16::from_f32(0.0);
    // let weights = [wout].into_iter().flatten().collect::<Vec<_>>();

    let input_buf = Arc::new(Buffer::create_mapped_storage(
        &device,
        bytemuck::cast_slice(&input),
    ));
    let weights_buf = Arc::new(Buffer::create_mapped_storage(
        &device,
        bytemuck::cast_slice(&weights),
    ));
    let output_buf = Arc::new(Buffer::create(
        &device,
        BufferInfo {
            size: (n_outputs * batch_size) * std::mem::size_of::<f16>(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuToCpu,
            ..Default::default()
        },
    ));

    let mut rgraph = RGraph::new();

    mlp_inference(
        &device,
        &mut rgraph,
        input_buf,
        weights_buf,
        output_buf.clone(),
        None,
        MlpConfig {
            output_stride: batch_size as _,
            batch_size: batch_size as _,
            in_width: 64,
            n_hidden_matmuls: hidden_layers,
            input_layout: 0,
            output_layout: 0,
        },
        batch_size,
        width,
        width,
        hidden_layers as _,
        width,
    );

    rgraph.submit(&device);

    let output: &[f16] = bytemuck::cast_slice(&output_buf.mapped_slice());
    let ouput_row0 = &output[0..64];
    let reference_row0 = &reference[0..64];
    println!("output={ouput_row0:?}");
    // println!("input={input_row0:?}");
    println!("reference={reference_row0:?}");
    // println!("reference={reference:?}");
}
