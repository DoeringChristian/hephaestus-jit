use ash::vk;
use std::sync::Arc;
use std::sync::Mutex;
use vk_sync::AccessType;

use super::vulkan_core::buffer::{Buffer, BufferInfo, MemoryLocation};
use super::vulkan_core::device::Device;
use super::vulkan_core::graph::RGraph;
use super::{vulkan_core::acceleration_structure::*, VulkanDevice};

use crate::backend::vulkan::vulkan_core::pipeline::{
    Binding, BufferWriteInfo, DescSetLayout, PipelineInfo, WriteSet,
};
use crate::backend::{AccelDesc, GeometryDesc};

pub enum AccelGeometryBuildInfo<'a> {
    Triangles {
        triangles: &'a Arc<Buffer>,
        vertices: &'a Arc<Buffer>,
    },
}
pub struct AccelBuildInfo<'a> {
    pub geometries: &'a [AccelGeometryBuildInfo<'a>],
    pub instances: &'a Arc<Buffer>,
}

#[derive(Debug)]
pub struct Accel {
    pub device: VulkanDevice,
    pub blases: Vec<Arc<AccelerationStructure>>,
    pub tlas: Arc<AccelerationStructure>,
    pub instance_buffer: Arc<Buffer>,
    pub info: AccelDesc,
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
            .map(|blas| Arc::new(blas))
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
        let tlas = Arc::new(AccelerationStructure::create(device, create_info));

        let instance_buffer = Buffer::create(
            &device,
            BufferInfo {
                size: info.instances * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_location: MemoryLocation::GpuOnly,
                ..Default::default()
            },
        );

        Self {
            blases,
            tlas,
            device: device.clone(),
            // instance_buffer: Mutex::new(instance_buffer),
            instance_buffer: Arc::new(instance_buffer),
            info: info.clone(),
        }
    }
    pub fn build(&self, rgraph: &mut RGraph, desc: AccelBuildInfo) {
        for (i, blas) in self.blases.iter().enumerate() {
            let mut info = blas.info.clone();
            assert_eq!(info.geometries.len(), 1);

            let mut deps = vec![];

            match &mut info.geometries[0].data {
                AccelerationStructureGeometryData::Triangles {
                    index_data,
                    vertex_data,
                    ..
                } => match desc.geometries[i] {
                    AccelGeometryBuildInfo::Triangles {
                        triangles,
                        vertices,
                    } => {
                        *index_data =
                            DeviceOrHostAddress::DeviceAddress(triangles.device_address());
                        *vertex_data =
                            DeviceOrHostAddress::DeviceAddress(vertices.device_address());
                        deps.push(triangles.clone());
                        deps.push(vertices.clone());
                    }
                },
                _ => todo!(),
            }

            blas.build(rgraph, &info, &[], &deps);
        }
        // let memory_barriers = [vk::MemoryBarrier::builder()
        //     .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
        //     .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)
        //     .build()];

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
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(blas.accel),
                    )
            })
            .collect::<Vec<_>>();

        let mut references_buffer = Buffer::create(
            &self.device,
            BufferInfo {
                size: std::mem::size_of::<u64>() * references.len(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                memory_location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
        );

        references_buffer
            .mapped_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&references));
        let references_buffer = Arc::new(references_buffer);

        let copy2instances = self.device.get_pipeline(&PipelineInfo {
            code: inline_spirv::include_spirv!(
                "src/backend/vulkan/builtin/kernels/copy2instances.glsl",
                comp
            ),
            desc_set_layouts: &[DescSetLayout {
                bindings: &[
                    Binding {
                        binding: 0,
                        count: 1,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    Binding {
                        binding: 1,
                        count: 1,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    Binding {
                        binding: 2,
                        count: 1,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                    },
                ],
            }],
        });

        {
            let n_instances = self.info.instances;
            let desc_instance_buffer = desc.instances.clone();
            let instance_buffer = self.instance_buffer.clone();
            rgraph
                .pass("Create VkAccelerationStructureInstanceKHR")
                .read(&references_buffer, AccessType::ComputeShaderReadOther)
                .read(&desc_instance_buffer, AccessType::ComputeShaderReadOther)
                .write(&instance_buffer, AccessType::ComputeShaderWrite)
                .record(move |device, cb, pool| {
                    copy2instances.submit(
                        cb,
                        pool,
                        device,
                        &[WriteSet {
                            set: 0,
                            binding: 0,
                            buffers: &[
                                BufferWriteInfo {
                                    buffer: &desc_instance_buffer,
                                },
                                BufferWriteInfo {
                                    buffer: &references_buffer,
                                },
                                BufferWriteInfo {
                                    buffer: &instance_buffer,
                                },
                            ],
                        }],
                        (n_instances as _, 1, 1),
                    );
                });
        }

        let mut info = self.tlas.info.clone();
        assert_eq!(info.geometries.len(), 1);

        match &mut info.geometries[0].data {
            AccelerationStructureGeometryData::Instances {
                array_of_pointers,
                data,
            } => {
                *data = DeviceOrHostAddress::DeviceAddress(self.instance_buffer.device_address());
            }
            _ => todo!(),
        }

        self.tlas.build(
            rgraph,
            &info,
            &self.blases,
            std::slice::from_ref(&self.instance_buffer),
        );
    }
    pub fn get_blas_device_address(&self, id: usize) -> vk::DeviceAddress {
        unsafe {
            self.device
                .acceleration_structure_ext
                .as_ref()
                .unwrap()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(self.blases[id].accel),
                )
        }
    }
    pub fn get_tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas.accel
    }
}
