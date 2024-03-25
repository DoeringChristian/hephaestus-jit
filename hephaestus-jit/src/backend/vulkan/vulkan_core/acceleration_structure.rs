use std::sync::Arc;

use super::buffer::{Buffer, BufferInfo, MemoryLocation};
use super::device::Device;
use super::graph::RGraph;
use ash::vk;
use vk_sync::AccessType;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceOrHostAddress {
    DeviceAddress(vk::DeviceAddress),

    HostAddress,
}
impl DeviceOrHostAddress {
    pub fn null() -> Self {
        Self::DeviceAddress(0)
    }
    pub fn to_vk(self) -> vk::DeviceOrHostAddressConstKHR {
        match self {
            DeviceOrHostAddress::DeviceAddress(device_address) => {
                vk::DeviceOrHostAddressConstKHR { device_address }
            }
            DeviceOrHostAddress::HostAddress => todo!(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AccelerationStructureInfo {
    pub ty: vk::AccelerationStructureTypeKHR,

    pub flags: vk::BuildAccelerationStructureFlagsKHR,

    pub geometries: Vec<AccelerationStructureGeometry>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AccelerationStructureGeometry {
    pub max_primitive_count: u32,

    pub flags: vk::GeometryFlagsKHR,

    pub data: AccelerationStructureGeometryData,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccelerationStructureGeometryData {
    AABBs {
        stride: vk::DeviceSize,
    },

    Instances {
        array_of_pointers: bool,

        data: DeviceOrHostAddress,
    },

    Triangles {
        index_data: DeviceOrHostAddress,

        index_type: vk::IndexType,

        max_vertex: u32,

        transform_data: DeviceOrHostAddress,

        vertex_data: DeviceOrHostAddress,

        vertex_format: vk::Format,

        vertex_stride: vk::DeviceSize,
    },
}
impl AccelerationStructureGeometryData {
    pub fn to_vulkan(
        self,
    ) -> (
        vk::GeometryTypeKHR,
        vk::AccelerationStructureGeometryDataKHR<'static>,
    ) {
        match self {
            AccelerationStructureGeometryData::AABBs { stride } => (
                vk::GeometryTypeKHR::AABBS,
                vk::AccelerationStructureGeometryDataKHR {
                    aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                        stride,
                        ..Default::default()
                    },
                },
            ),
            AccelerationStructureGeometryData::Instances {
                array_of_pointers,
                data,
            } => (
                vk::GeometryTypeKHR::INSTANCES,
                vk::AccelerationStructureGeometryDataKHR {
                    instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                        array_of_pointers: array_of_pointers as _,
                        data: data.to_vk(),
                        ..Default::default()
                    },
                },
            ),
            AccelerationStructureGeometryData::Triangles {
                index_data,
                index_type,
                max_vertex,
                transform_data,
                vertex_data,
                vertex_format,
                vertex_stride,
            } => (
                vk::GeometryTypeKHR::TRIANGLES,
                vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                        vertex_format,
                        vertex_stride,
                        max_vertex,
                        index_type,
                        transform_data: transform_data.to_vk(),
                        vertex_data: vertex_data.to_vk(),
                        index_data: index_data.to_vk(),
                        ..Default::default()
                    },
                },
            ),
        }
    }
}

#[derive(Debug)]
/// General AccelerationStructure Abstraction over vk::AccelerationstructureKHR
///
/// * `buffer`: Buffer storing the actual data of the AccelerationStructure
/// * `accel`: Vulkan AccelerationStructure
/// * `device`: Device
/// * `sizes`: Various sizes for scratch buffer etc
/// * `info`: The info struct with which the AccelerationStructure was created
pub struct AccelerationStructure {
    pub(crate) buffer: Buffer, // Buffer to store AccelerationStructure
    pub accel: vk::AccelerationStructureKHR,

    device: Arc<Device>,

    sizes: vk::AccelerationStructureBuildSizesInfoKHR<'static>,
    pub info: AccelerationStructureInfo,
}
unsafe impl Send for AccelerationStructure {}
unsafe impl Sync for AccelerationStructure {}
impl AccelerationStructure {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn create(device: &Arc<Device>, info: AccelerationStructureInfo) -> Self {
        log::trace!("Creating AccelerationStructure with {info:#?}");
        let (geometries, max_primitive_counts): (Vec<_>, Vec<_>) = info
            .geometries
            .iter()
            .map(|info| {
                let (geometry_type, geometry) = info.data.to_vulkan();
                (
                    vk::AccelerationStructureGeometryKHR {
                        geometry_type,
                        geometry,
                        flags: info.flags,
                        ..Default::default()
                    },
                    info.max_primitive_count,
                )
            })
            .unzip();

        let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(info.ty)
            .flags(info.flags)
            .geometries(&geometries);

        let sizes = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &geometry_info,
                    &max_primitive_counts,
                    &mut size_info,
                );
            size_info
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
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(info.ty)
            .buffer(buffer.vk())
            .size(sizes.acceleration_structure_size);

        let accel = unsafe {
            device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .create_acceleration_structure(&create_info, None)
                .unwrap()
        };

        log::trace!(
            "Created AccelerationStructure with handle {handle:?}",
            handle = accel
        );
        Self {
            buffer,
            accel,
            sizes,
            device: device.clone(),
            info,
        }
    }
    /// Build the acceleration structure
    ///
    /// * `rgraph`: The current render graph
    /// * `info`: AccelerationStructureInfo
    /// * `dependencies`: dependencies of this acceleration structure, used to resolve render graph
    /// * `buffer_dependencies`: dependencies of this acceleration structure, used to resolve render graph
    /// TODO: somehow remove dependencies and buffer dependencies by submitting them through the
    /// info struct
    pub fn build(
        self: &Arc<Self>,
        rgraph: &mut RGraph,
        info: &AccelerationStructureInfo,
        // TODO: not shure if I should handle dependencies like this:
        dependencies: &[Arc<AccelerationStructure>],
        buffer_dependencies: &[Arc<Buffer>],
    ) {
        log::trace!("Building AccelerationStructure with {info:#?}");
        // TODO: pool
        let scratch_buffer = Arc::new(Buffer::create(
            &self.device,
            BufferInfo {
                size: self.sizes.build_scratch_size as _,
                alignment: self
                    .device
                    .physical_device
                    .acceleration_structure_properties
                    .min_acceleration_structure_scratch_offset_alignment
                    as _,
                usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::GpuOnly,
            },
        ));
        let scratch_buffer_address = scratch_buffer.device_address();

        let (geometries, build_ranges): (Vec<_>, Vec<_>) = info
            .geometries
            .iter()
            .map(|info| {
                let (geometry_type, geometry) = info.data.to_vulkan();
                (
                    vk::AccelerationStructureGeometryKHR {
                        geometry_type,
                        geometry,
                        flags: info.flags,
                        ..Default::default()
                    },
                    vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: info.max_primitive_count,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    },
                )
            })
            .unzip();

        // log::trace!("Using geometry info {geometry_info:#?}");

        let s = self.clone();
        let mut pass = rgraph
            .pass("Build AccelerationStructure")
            .read(&scratch_buffer, AccessType::AccelerationStructureBuildRead)
            .write(&scratch_buffer, AccessType::AccelerationStructureBuildWrite);
        for dep in dependencies {
            pass = pass.read(dep, AccessType::AccelerationStructureBuildRead);
        }
        for dep in buffer_dependencies {
            pass = pass.read(dep, AccessType::AccelerationStructureBuildRead);
        }
        pass.write(self, AccessType::AccelerationStructureBuildWrite)
            .record(move |device, cb, _| unsafe {
                // We need to construct geometry_info here, because vulkan uses pointers
                let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(s.info.ty)
                    .flags(s.info.flags)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(s.accel)
                    .geometries(&geometries)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffer_address,
                    });

                device
                    .acceleration_structure_ext
                    .as_ref()
                    .unwrap()
                    .cmd_build_acceleration_structures(cb, &[geometry_info], &[&build_ranges]);
            });

        log::trace!(
            "Built AccelerationStructure with handle {handle:?}",
            handle = self.accel
        );
    }
    pub fn info(&self) -> &AccelerationStructureInfo {
        &self.info
    }
}

impl Drop for AccelerationStructure {
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
