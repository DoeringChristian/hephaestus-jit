#[cfg(test)]
use std::path::Path;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use half::f16;
use num_traits::Float;

use crate::backend::vulkan::vulkan_core::buffer::{Buffer, BufferInfo};
use crate::backend::vulkan::vulkan_core::graph::RGraph;
use crate::backend::vulkan::VulkanDevice;

use super::super::fused_mlp::*;
#[test]
fn fused_mlp_inference() {
    pretty_env_logger::try_init().ok();
    let device = VulkanDevice::create(0).unwrap();

    // input: [batch_size x width, RM] = HT
    // weight: [width x width, RM] = W
    // output: [batch_size x width, RM] = H'T
    //    H' = W @ H
    // => H'T = HT @ WT
    //
    // The algorithm loads W transposed by interpreting it in column major order
    // [128x64 RM] -> [64x64] -> ( _/ ) -> [64x64] -> [64]
    let n_inputs = 64;
    let n_outputs = 64;
    let hidden_layers = 2;
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
            batch_size: batch_size as _,
            hidden_layers,
        },
        batch_size,
        width,
        width,
        width,
    );

    rgraph.submit(&device);

    let output: &[f16] = bytemuck::cast_slice(&output_buf.mapped_slice());

    let row = 100;
    let row_range = (row * 64)..(row + 1) * 64;

    let ouput_row = &output[row_range.clone()];
    let reference_row = &reference[row_range.clone()];

    println!("output={ouput_row:?}");
    println!("reference={reference_row:?}");

    // println!("output={output:?}");
    // println!("reference={reference:?}");

    dbg!(output
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .reduce(|a, b| a.max(b)));
    let mean = output
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (a.to_f32() - b.to_f32()).powi(2))
        .reduce(|a, b| a + b)
        .unwrap()
        / output.len() as f32;
    dbg!(mean);
    assert!(
        output
            .iter()
            .zip(&reference)
            .all(|(&a, &b)| (a - b).abs() < f16::from_f32(10.0)),
        "lhs = {output:?}\n is not equal to rhs = {reference:?}\n"
    );
}
