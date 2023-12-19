use crate::backend::vulkan::pipeline::{Binding, BufferWriteInfo, DescSetLayout, WriteSet};
use crate::backend::{self, Buffer};

#[test]
fn image() {
    let device = backend::Device::vulkan(0).unwrap();

    let tex = device.create_texture([100, 100, 100], 4).unwrap();
}

#[test]
fn prefix_sum() {
    use backend::vulkan::*;
    let device = VulkanDevice::create(0).unwrap();

    let elem_size = std::mem::size_of::<u32>();
    let num = 1024;

    let block_size = 128;
    let block_count = 8;
    let scratch_items = 32 + block_count;

    let mut pool = Pool::new(&device);

    // let prefix_sum_large_init = device.get_shader_glsl(
    //     include_str!("kernels/prefix_sum_large_init.glsl"),
    //     ShaderKind::Compute,
    //     &[("WORK_GROUP_SIZE", Some(&format!("{block_size}")))],
    // );
    // let prefix_sum_large_init = device.get_pipeline(&PipelineDesc {
    //     code: &prefix_sum_large_init,
    //     desc_set_layouts: &[DescSetLayout {
    //         bindings: &[
    //             Binding {
    //                 binding: 0,
    //                 count: 1,
    //             },
    //             Binding {
    //                 binding: 1,
    //                 count: 1,
    //             },
    //         ],
    //     }],
    // });

    let prefix_sum_large = device.get_shader_glsl(
        include_str!("kernels/prefix_sum_large.glsl"),
        ShaderKind::Compute,
        &[("WORK_GROUP_SIZE", Some(&format!("{block_size}")))],
    );

    let pipeline = device.get_pipeline(&PipelineDesc {
        code: &prefix_sum_large,
        desc_set_layouts: &[DescSetLayout {
            bindings: &(0..5)
                .map(|i| Binding {
                    binding: i,
                    count: 1,
                })
                .collect::<Vec<_>>(),
        }],
    });

    let mut output = pool.buffer(BufferInfo {
        size: num * elem_size,
        alignment: 0,
        usage: vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        memory_location: MemoryLocation::CpuToGpu,
    });
    output
        .mapped_slice_mut()
        .copy_from_slice(bytemuck::cast_slice(&vec![0u32; num]));
    let mut input = pool.buffer(BufferInfo {
        size: num * elem_size,
        alignment: 0,
        usage: vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        memory_location: MemoryLocation::CpuToGpu,
    });
    input
        .mapped_slice_mut()
        .copy_from_slice(bytemuck::cast_slice(&vec![1u32; num]));
    let mut size = pool.buffer(BufferInfo {
        size: std::mem::size_of::<u32>(),
        alignment: 0,
        usage: vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        memory_location: MemoryLocation::CpuToGpu,
    });
    size.mapped_slice_mut()
        .copy_from_slice(bytemuck::cast_slice(&[num as u32]));
    let mut index = pool.buffer(BufferInfo {
        size: std::mem::size_of::<u32>(),
        alignment: 0,
        usage: vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        memory_location: MemoryLocation::CpuToGpu,
    });
    index
        .mapped_slice_mut()
        .copy_from_slice(bytemuck::cast_slice(&[0]));
    let mut scratch = pool.buffer(BufferInfo {
        size: scratch_items * std::mem::size_of::<u64>(),
        alignment: 0,
        usage: vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        memory_location: MemoryLocation::CpuToGpu,
    });
    scratch
        .mapped_slice_mut()
        .copy_from_slice(bytemuck::cast_slice(
            &(0..32)
                .map(|_| 2)
                .chain((0..block_count).map(|_| 0))
                .collect::<Vec<u64>>(),
        ));

    // let s: &[u64] = bytemuck::cast_slice(scratch.mapped_slice());
    // dbg!(s);
    //
    // let out: &[u32] = bytemuck::cast_slice(output.mapped_slice());
    // dbg!(out);

    device.submit_global(|device, cb| {
        pipeline.submit(
            cb,
            &mut pool,
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
                    buffers: &[BufferWriteInfo { buffer: &output }],
                },
                WriteSet {
                    set: 0,
                    binding: 2,
                    buffers: &[BufferWriteInfo { buffer: &size }],
                },
                WriteSet {
                    set: 0,
                    binding: 3,
                    buffers: &[BufferWriteInfo { buffer: &scratch }],
                },
                WriteSet {
                    set: 0,
                    binding: 4,
                    buffers: &[BufferWriteInfo { buffer: &index }],
                },
            ],
            (block_count as _, 1, 1),
        );
    });

    let out: &[u32] = bytemuck::cast_slice(output.mapped_slice());
    for i in 0..(block_count as usize - 1) {
        dbg!(&out[i * block_size..(i + 1) * block_size]);
    }
}
