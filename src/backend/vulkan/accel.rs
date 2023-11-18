use super::buffer::{Buffer, BufferInfo, MemoryLocation};
use super::context::Context;
use super::device::Device;
use crate::backend::{AccelDesc, GeometryDesc};
use ash::vk;

// pub enum GeometryCreateDesc {
//     Triangles {
//         n_triangles: usize,
//         n_vertices: usize,
//     },
// }
//
// pub struct AccelCreateDesc<'a> {
//     pub geometries: &'a [GeometryCreateDesc],
//     pub instances: usize,
// }

pub enum GeometryBuildDesc<'a> {
    Triangles {
        triangles: &'a Buffer,
        vertices: &'a Buffer,
    },
}
// pub enum InstanceBuildDesc {}

pub struct AccelBuildDesc<'a> {
    pub geometries: &'a [GeometryBuildDesc<'a>],
    // pub instances: &'a [InstanceBuildDesc],
    pub instances: Buffer,
}

#[derive(Debug)]
pub struct AccelInternal {
    buffer: Buffer,
    accel: vk::AccelerationStructureKHR,
    sizes: vk::AccelerationStructureBuildSizesInfoKHR,

    device: Device,
    n_primitives: usize,
    geometry_type: vk::GeometryTypeKHR,
}
impl AccelInternal {
    pub fn create_tlas(device: &Device, desc: &AccelDesc) -> Self {
        let geometry = vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                array_of_pointers: vk::FALSE,
                data: vk::DeviceOrHostAddressConstKHR { device_address: 0 },
                ..Default::default()
            },
        };
        let geometry_type = vk::GeometryTypeKHR::INSTANCES;
        let n_primitives = desc.instances.len();

        let geometries = [vk::AccelerationStructureGeometryKHR {
            geometry_type,
            geometry,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            ..Default::default()
        }];

        let info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries);

        let sizes = unsafe {
            device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &info,
                    &[n_primitives as u32],
                )
        };
        let buffer = Buffer::create(
            device,
            BufferInfo {
                size: sizes.acceleration_structure_size as _,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_location: MemoryLocation::GpuOnly,
                ..Default::default()
            },
        );
        let info = vk::AccelerationStructureCreateInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .buffer(buffer.buffer())
            .size(sizes.acceleration_structure_size);

        let accel = unsafe {
            device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .create_acceleration_structure(&info, None)
                .unwrap()
        };
        Self {
            buffer,
            accel,
            sizes,
            device: device.clone(),
            n_primitives,
            geometry_type,
        }
    }
    pub fn create_blas(device: &Device, desc: &GeometryDesc) -> Self {
        // Get Geometries for sizes
        let (geometry_type, geometry, n_primitives) = match desc {
            GeometryDesc::Triangles {
                n_triangles,
                n_vertices,
            } => (
                vk::GeometryTypeKHR::TRIANGLES,
                vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                        vertex_format: vk::Format::R32G32B32_SFLOAT,
                        vertex_stride: 12,
                        max_vertex: *n_vertices as _,
                        index_type: vk::IndexType::UINT32,
                        transform_data: vk::DeviceOrHostAddressConstKHR { device_address: 0 },
                        ..Default::default()
                    },
                },
                *n_triangles,
            ),
        };
        let geometries = [vk::AccelerationStructureGeometryKHR {
            geometry_type,
            geometry,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            ..Default::default()
        }];

        let info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries);

        let sizes = unsafe {
            device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &info,
                    &[n_primitives as u32],
                )
        };
        let buffer = Buffer::create(
            device,
            BufferInfo {
                size: sizes.acceleration_structure_size as _,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_location: MemoryLocation::GpuOnly,
                ..Default::default()
            },
        );
        let info = vk::AccelerationStructureCreateInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .buffer(buffer.buffer())
            .size(sizes.acceleration_structure_size);

        let accel = unsafe {
            device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .create_acceleration_structure(&info, None)
                .unwrap()
        };
        Self {
            buffer,
            accel,
            sizes,
            device: device.clone(),
            n_primitives,
            geometry_type,
        }
    }
    pub fn build_tlas(&self, ctx: &mut Context, desc: &AccelBuildDesc) {
        let scratch_buffer = ctx.buffer(BufferInfo {
            size: self.sizes.build_scratch_size as _,
            usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        });
        let scratch_buffer = scratch_buffer.device_address();

        let geometry_type = vk::GeometryTypeKHR::INSTANCES;
        let geometry = vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                array_of_pointers: vk::FALSE,
                data: vk::DeviceOrHostAddressConstKHR {
                    device_address: desc.instances.device_address(),
                },
                ..Default::default()
            },
        };

        let geometries = [vk::AccelerationStructureGeometryKHR {
            geometry_type,
            geometry,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            ..Default::default()
        }];

        unsafe {
            ctx.acceleration_structure_ext
                .as_ref()
                .unwrap()
                .cmd_build_acceleration_structures(
                    ctx.cb,
                    &[vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .dst_acceleration_structure(self.accel)
                        .geometries(&geometries)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch_buffer,
                        })
                        .build()],
                    &[&[vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: self.n_primitives as _,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    }]],
                );
        }
    }
    pub fn build_blas(&self, ctx: &mut Context, desc: &GeometryBuildDesc) {
        let scratch_buffer = ctx.buffer(BufferInfo {
            size: self.sizes.build_scratch_size as _,
            usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
            ..Default::default()
        });
        let scratch_buffer = scratch_buffer.device_address();

        let (geometry_type, geometry) = match desc {
            GeometryBuildDesc::Triangles {
                triangles,
                vertices,
            } => (
                vk::GeometryTypeKHR::TRIANGLES,
                vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                        vertex_format: vk::Format::R32G32B32_SFLOAT,
                        vertex_stride: 12,
                        vertex_data: vk::DeviceOrHostAddressConstKHR {
                            device_address: vertices.device_address(),
                        },
                        max_vertex: (vertices.info().size / 3) as _,
                        index_type: vk::IndexType::UINT32,
                        index_data: vk::DeviceOrHostAddressConstKHR {
                            device_address: triangles.device_address(),
                        },
                        transform_data: vk::DeviceOrHostAddressConstKHR { device_address: 0 },
                        ..Default::default()
                    },
                },
            ),
        };
        let geometries = [vk::AccelerationStructureGeometryKHR {
            geometry_type,
            geometry,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            ..Default::default()
        }];
        unsafe {
            ctx.acceleration_structure_ext
                .as_ref()
                .unwrap()
                .cmd_build_acceleration_structures(
                    ctx.cb,
                    &[vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .dst_acceleration_structure(self.accel)
                        .geometries(&geometries)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch_buffer,
                        })
                        .build()],
                    &[&[vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: self.n_primitives as _,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    }]],
                );
        }
    }
}

impl Drop for AccelInternal {
    fn drop(&mut self) {
        unsafe {
            self.device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .destroy_acceleration_structure(self.accel, None);
        }
    }
}

#[derive(Debug)]
pub struct Accel {
    device: Device,
    blases: Vec<AccelInternal>,
    tlas: AccelInternal,
}

impl Accel {
    pub fn create(device: &Device, desc: &AccelDesc) -> Self {
        let blases = desc
            .geometries
            .iter()
            .map(|desc| AccelInternal::create_blas(device, desc))
            .collect::<Vec<_>>();
        let tlas = AccelInternal::create_tlas(device, &desc);

        Self {
            blases,
            tlas,
            device: device.clone(),
        }
    }
    pub fn build(&self, ctx: &mut Context, desc: &AccelBuildDesc) {
        // TODO: Heavy validation effort
        for (i, blas) in self.blases.iter().enumerate() {
            blas.build_blas(ctx, &desc.geometries[i]);
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

        self.tlas.build_tlas(ctx, desc);
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
}
