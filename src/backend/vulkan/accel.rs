use ash::vk;
use std::sync::Mutex;

use super::buffer::{Buffer, BufferInfo, MemoryLocation};
use super::context::Context;
use super::device::Device;
use super::{acceleration_structure::*, VulkanDevice};

use crate::backend::vulkan::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineDesc, WriteSet,
};
use crate::backend::{AccelDesc, GeometryDesc};

pub enum AccelGeometryBuildInfo<'a> {
    Triangles {
        triangles: &'a Buffer,
        vertices: &'a Buffer,
    },
}
pub struct AccelBuildInfo<'a> {
    pub geometries: &'a [AccelGeometryBuildInfo<'a>],
    pub instances: &'a Buffer,
}

#[derive(Debug)]
pub struct Accel {
    device: VulkanDevice,
    blases: Vec<AccelerationStructure>,
    tlas: AccelerationStructure,
    instance_buffer: Mutex<Buffer>,
    info: AccelDesc,
}

impl Accel {
    pub fn create(device: &VulkanDevice, info: &AccelDesc) -> Self {
        let blases = info
            .geometries
            .iter()
            .map(|desc| match desc {
                GeometryDesc::Triangles {
                    n_triangles,
                    n_vertices,
                } => {
                    let info = AccelerationStructureInfo {
                        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                        flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
                        geometries: vec![AccelerationStructureGeometry {
                            max_primitive_count: *n_triangles as _,
                            flags: vk::GeometryFlagsKHR::empty(),
                            data: AccelerationStructureGeometryData::Triangles {
                                index_data: DeviceOrHostAddress::null(),
                                index_type: vk::IndexType::UINT32,
                                max_vertex: *n_triangles as _,
                                transform_data: DeviceOrHostAddress::null(),
                                vertex_data: DeviceOrHostAddress::null(),
                                vertex_format: vk::Format::R32G32B32_SFLOAT,
                                vertex_stride: 12,
                            },
                        }],
                    };
                    AccelerationStructure::create(device, info)
                }
            })
            .collect::<Vec<_>>();

        let create_info = AccelerationStructureInfo {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
            geometries: vec![AccelerationStructureGeometry {
                max_primitive_count: info.instances as _,
                flags: vk::GeometryFlagsKHR::empty(),
                data: AccelerationStructureGeometryData::Instances {
                    array_of_pointers: false,
                    data: DeviceOrHostAddress::null(),
                },
            }],
        };
        let tlas = AccelerationStructure::create(device, create_info);

        let instance_buffer = Buffer::create(
            &device,
            BufferInfo {
                size: info.instances * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
        );

        Self {
            blases,
            tlas,
            device: device.clone(),
            instance_buffer: Mutex::new(instance_buffer),
            info: info.clone(),
        }
    }
    pub fn build(&self, ctx: &mut Context, desc: AccelBuildInfo) {
        for (i, blas) in self.blases.iter().enumerate() {
            let mut info = blas.info.clone();
            assert_eq!(info.geometries.len(), 1);

            match &mut info.geometries[0].data {
                AccelerationStructureGeometryData::Triangles {
                    index_data,
                    index_type,
                    max_vertex,
                    transform_data,
                    vertex_data,
                    vertex_format,
                    vertex_stride,
                } => match desc.geometries[i] {
                    AccelGeometryBuildInfo::Triangles {
                        triangles,
                        vertices,
                    } => {
                        *index_data =
                            DeviceOrHostAddress::DeviceAddress(triangles.device_address());
                        *vertex_data =
                            DeviceOrHostAddress::DeviceAddress(vertices.device_address());
                    }
                },
                _ => todo!(),
            }

            blas.build(ctx, &info);
        }
        let memory_barriers = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)
            .build()];
        unsafe {
            ctx.cmd_pipeline_barrier(
                ctx.cb,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &memory_barriers,
                &[],
                &[],
            );
        }

        // Create references buffer
        let references = self
            .blases
            .iter()
            .map(|blas| unsafe {
                self.device
                    .acceleration_structure_ext
                    .as_ref()
                    .unwrap()
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                            .acceleration_structure(blas.accel),
                    )
            })
            .collect::<Vec<_>>();

        let cb = ctx.cb;

        let references_buffer = ctx.buffer_mut(BufferInfo {
            size: std::mem::size_of::<u64>() * references.len(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
            ..Default::default()
        });

        references_buffer
            .mapped_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&references));

        let copy2instances = self.device.get_pipeline(&PipelineDesc {
            code: inline_spirv::include_spirv!(
                "src/backend/vulkan/kernels/copy2instances.glsl",
                comp
            ),
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
                    Binding {
                        binding: 2,
                        count: 1,
                    },
                ],
            }],
        });

        let instance_buffer = self.instance_buffer.lock().unwrap();

        copy2instances.submit(
            cb,
            &self.device,
            &[WriteSet {
                set: 0,
                binding: 0,
                buffers: &[
                    BufferWriteInfo {
                        buffer: &desc.instances,
                    },
                    BufferWriteInfo {
                        buffer: &references_buffer,
                    },
                    BufferWriteInfo {
                        buffer: &instance_buffer,
                    },
                ],
            }],
            (self.info.instances as _, 1, 1),
        );

        // Memory Barrierr
        let memory_barriers = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build()];
        unsafe {
            ctx.cmd_pipeline_barrier(
                ctx.cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &memory_barriers,
                &[],
                &[],
            );
        }

        let mut info = self.tlas.info.clone();
        assert_eq!(info.geometries.len(), 1);

        match &mut info.geometries[0].data {
            AccelerationStructureGeometryData::Instances {
                array_of_pointers,
                data,
            } => {
                *data = DeviceOrHostAddress::DeviceAddress(instance_buffer.device_address());
            }
            _ => todo!(),
        }

        self.tlas.build(ctx, &info);
    }
    pub fn get_blas_device_address(&self, id: usize) -> vk::DeviceAddress {
        unsafe {
            self.device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                        .acceleration_structure(self.blases[id].accel),
                )
        }
    }
    pub fn get_tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas.accel
    }
}
