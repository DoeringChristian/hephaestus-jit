use crate::backend::vulkan::pipeline::{Binding, BufferWriteInfo, DescSetLayout, WriteSet};
use crate::backend::{self, Buffer, TextureDesc};
use crate::vartype::AsVarType;
use crate::vartype::VarType;
use crate::vulkan;

#[test]
fn image() {
    let device = backend::Device::vulkan(0).unwrap();

    let tex = device
        .create_texture(&TextureDesc {
            shape: [100, 100, 100],
            channels: 4,
            format: f32::var_ty(),
        })
        .unwrap();
}

// // #[test]
// fn compress_large() {
//     use backend::vulkan::*;
//     let device = VulkanDevice::create(0).unwrap();
//
//     let num = 2048 * 4;
//
//     let items_per_thread = 16;
//     let block_size = 128;
//     let warp_size = device.physical_device.subgroup_properties.subgroup_size as usize;
//     let items_per_block = items_per_thread * block_size;
//     let block_count = (num + items_per_block - 1) / items_per_block;
//     let scratch_items = 1 + warp_size + block_count;
//
//     let mut pool = Pool::new(&device);
//
//     let mut output = pool.buffer(BufferInfo {
//         size: num,
//         alignment: 0,
//         usage: vk::BufferUsageFlags::TRANSFER_SRC
//             | vk::BufferUsageFlags::TRANSFER_DST
//             | vk::BufferUsageFlags::STORAGE_BUFFER,
//         memory_location: MemoryLocation::GpuToCpu,
//     });
//
//     let mut output_count = pool.buffer(BufferInfo {
//         size: 4,
//         alignment: 0,
//         usage: vk::BufferUsageFlags::TRANSFER_SRC
//             | vk::BufferUsageFlags::TRANSFER_DST
//             | vk::BufferUsageFlags::STORAGE_BUFFER,
//         memory_location: MemoryLocation::GpuToCpu,
//     });
//
//     let input_vec = (0..num).map(|i| 1u8).collect::<Vec<_>>();
//
//     let mut input = pool.buffer(BufferInfo {
//         size: num,
//         alignment: 0,
//         usage: vk::BufferUsageFlags::TRANSFER_SRC
//             | vk::BufferUsageFlags::TRANSFER_DST
//             | vk::BufferUsageFlags::STORAGE_BUFFER,
//         memory_location: MemoryLocation::CpuToGpu,
//     });
//     input
//         .mapped_slice_mut()
//         .copy_from_slice(bytemuck::cast_slice(&input_vec));
//
//     // let mut scratch_buffer = None;
//     device.submit_global(|_, cb| {
//         device.compress_large(cb, &mut pool, num, &output_count, &input, &output);
//     });
//
//     let out: &[u32] = bytemuck::cast_slice(output.mapped_slice());
//     for i in 0..(block_count as usize) {
//         println!("");
//         let slice = &out[i * block_size..(i + 1) * block_size];
//         println!("{slice:?}");
//     }
// }
// // #[test]
// fn prefix_sum() {
//     use backend::vulkan::*;
//     let device = VulkanDevice::create(0).unwrap();
//
//     let elem_size = std::mem::size_of::<u32>();
//     let num = 2048 * 4;
//
//     let items_per_thread = 4 * 4;
//     let block_size = 128;
//     let warp_size = device.physical_device.subgroup_properties.subgroup_size as usize;
//     let items_per_block = items_per_thread * block_size;
//     let block_count = (num + items_per_block - 1) / items_per_block;
//     let scratch_items = 1 + warp_size + block_count;
//     dbg!(block_count);
//
//     let mut pool = Pool::new(&device);
//
//     let mut output = pool.buffer(BufferInfo {
//         size: num * elem_size,
//         alignment: 0,
//         usage: vk::BufferUsageFlags::TRANSFER_SRC
//             | vk::BufferUsageFlags::TRANSFER_DST
//             | vk::BufferUsageFlags::STORAGE_BUFFER,
//         memory_location: MemoryLocation::CpuToGpu,
//     });
//     output
//         .mapped_slice_mut()
//         .copy_from_slice(bytemuck::cast_slice(&vec![0u32; num]));
//
//     let input_vec = (0..num as u32).map(|i| i).collect::<Vec<_>>();
//
//     let mut input = pool.buffer(BufferInfo {
//         size: num * elem_size,
//         alignment: 0,
//         usage: vk::BufferUsageFlags::TRANSFER_SRC
//             | vk::BufferUsageFlags::TRANSFER_DST
//             | vk::BufferUsageFlags::STORAGE_BUFFER,
//         memory_location: MemoryLocation::CpuToGpu,
//     });
//     input
//         .mapped_slice_mut()
//         .copy_from_slice(bytemuck::cast_slice(&input_vec));
//
//     // let mut scratch_buffer = None;
//     device.submit_global(|_, cb| {
//         // scratch_buffer = Some(device.prefix_sum_scratch_buffer(&mut pool, cb, scratch_items));
//         device.prefix_sum(cb, &mut pool, &VarType::U32, num, true, &input, &output);
//     });
//
//     // let scratch_buffer: &[u64] =
//     //     bytemuck::cast_slice(scratch_buffer.as_ref().unwrap().mapped_slice());
//     // dbg!(scratch_buffer);
//
//     let out: &[u32] = bytemuck::cast_slice(output.mapped_slice());
//     for i in 0..(block_count as usize) {
//         println!("");
//         let slice = &out[i * block_size..(i + 1) * block_size];
//         println!("{slice:?}");
//     }
//     let reference = input_vec
//         .into_iter()
//         .scan(0, |sum, i| {
//             *sum += i;
//             Some(*sum)
//         })
//         .collect::<Vec<_>>();
//
//     assert_eq!(out, &reference);
// }
