use ash::vk;
use std::sync::Mutex;

use super::acceleration_structure::*;
use super::buffer::{Buffer, BufferInfo, MemoryLocation};
use super::context::Context;
use super::device::Device;

use crate::backend::{AccelDesc, GeometryDesc, InstanceDesc};

pub enum AccelGeometryBuildInfo<'a> {
    Triangles {
        triangles: &'a Buffer,
        vertices: &'a Buffer,
    },
}
pub struct AccelBuildInfo<'a> {
    pub geometries: &'a [AccelGeometryBuildInfo<'a>],
    pub instances: &'a [InstanceDesc],
}

#[derive(Debug)]
pub struct Accel {
    device: Device,
    blases: Vec<AccelerationStructure>,
    tlas: AccelerationStructure,
    instance_buffer: Mutex<Buffer>,
}

impl Accel {
    pub fn create(device: &Device, desc: &AccelDesc) -> Self {
        let blases = desc
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

        let info = AccelerationStructureInfo {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
            geometries: vec![AccelerationStructureGeometry {
                max_primitive_count: desc.instances.len() as _,
                flags: vk::GeometryFlagsKHR::empty(),
                data: AccelerationStructureGeometryData::Instances {
                    array_of_pointers: false,
                    data: DeviceOrHostAddress::null(),
                },
            }],
        };
        let tlas = AccelerationStructure::create(device, info);

        let instance_buffer = Buffer::create(
            &device,
            BufferInfo {
                size: desc.instances.len()
                    * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
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
        let instances = desc
            .instances
            .iter()
            .map(|instance| vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: instance.transform,
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as _,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: self.get_blas_device_address(instance.geometry),
                },
            })
            .collect::<Vec<_>>();

        log::trace!(
            "Uploading {n} instances to instance buffer",
            n = instances.len()
        );

        let instance_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const _,
                std::mem::size_of_val(instances.as_slice()),
            )
        };
        let mut instance_buffer = self.instance_buffer.lock().unwrap();
        instance_buffer
            .mapped_slice_mut()
            .copy_from_slice(instance_slice);

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
