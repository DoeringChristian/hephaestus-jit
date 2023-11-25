mod cuda;
mod vulkan;
use std::fmt::Debug;

use cuda::{CudaBuffer, CudaDevice};
use vulkan::{VulkanBuffer, VulkanDevice, VulkanTexture};

use crate::graph::Graph;
use crate::ir::IR;
use crate::trace;
use crate::vartype::AsVarType;

use self::vulkan::VulkanAccel;

// TODO: Device Caching
pub fn vulkan(id: usize) -> Device {
    Device::vulkan(id).unwrap()
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("[CudaError] {0:?}")]
    CudaError(#[from] cuda::DriverError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
pub enum Device {
    CudaDevice(CudaDevice),
    VulkanDevice(VulkanDevice),
}

impl Device {
    pub fn cuda(id: usize) -> Result<Self> {
        Ok(Device::CudaDevice(CudaDevice::create(id)?))
    }
    pub fn vulkan(id: usize) -> Result<Self> {
        Ok(Device::VulkanDevice(VulkanDevice::create(id)?))
    }
    pub fn create_buffer(&self, size: usize) -> Result<Buffer> {
        let buffer = match self {
            Device::CudaDevice(device) => Buffer::CudaBuffer(device.create_buffer(size)?),
            Device::VulkanDevice(device) => Buffer::VulkanBuffer(device.create_buffer(size)?),
        };
        Ok(buffer)
    }
    pub fn create_buffer_from_slice(&self, slice: &[u8]) -> Result<Buffer> {
        Ok(match self {
            Device::CudaDevice(device) => {
                Buffer::CudaBuffer(device.create_buffer_from_slice(slice)?)
            }
            Device::VulkanDevice(device) => {
                Buffer::VulkanBuffer(device.create_buffer_from_slice(slice)?)
            }
        })
    }
    pub fn create_texture(&self, shape: [usize; 3], channels: usize) -> Result<Texture> {
        match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => Ok(Texture::VulkanTexture(
                device.create_texture(shape, channels)?,
            )),
        }
    }
    pub fn create_accel(&self, desc: &AccelDesc) -> Result<Accel> {
        match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => Ok(Accel::VulkanAccel(device.create_accel(desc)?)),
        }
    }
    pub fn execute_graph(&self, trace: &trace::Trace, graph: &Graph) -> Result<()> {
        match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => device.execute_graph(trace, graph),
        }
    }
    pub fn execute_ir(&self, ir: &IR, num: usize, buffers: &[&Buffer]) -> Result<()> {
        match self {
            Device::CudaDevice(device) => {
                let buffers = buffers
                    .iter()
                    .map(|b| match b {
                        Buffer::CudaBuffer(buffer) => buffer,
                        _ => todo!(),
                    })
                    .collect::<Vec<_>>();
                device.execute_ir(ir, num, &buffers)
            }
            Device::VulkanDevice(device) => {
                let buffers = buffers
                    .iter()
                    .map(|b| match b {
                        Buffer::VulkanBuffer(buffer) => buffer,
                        _ => todo!(),
                    })
                    .collect::<Vec<_>>();
                device.execute_ir(ir, num, &buffers)
            }
        }
    }
}

// TODO: we might not need it to be cloneable
#[derive(Debug)]
pub enum Buffer {
    CudaBuffer(CudaBuffer),
    VulkanBuffer(VulkanBuffer),
}
impl Buffer {
    pub fn size(&self) -> usize {
        match self {
            Buffer::CudaBuffer(buffer) => buffer.size(),
            Buffer::VulkanBuffer(buffer) => buffer.size(),
        }
    }
    pub fn to_host<T: bytemuck::Pod + AsVarType>(&self) -> Result<Vec<T>> {
        match self {
            Self::CudaBuffer(buffer) => Ok(buffer.to_host()?),
            Self::VulkanBuffer(buffer) => Ok(buffer.to_host()?),
        }
    }
    pub fn device(&self) -> Device {
        match self {
            Buffer::CudaBuffer(buffer) => Device::CudaDevice(buffer.device().clone()),
            Buffer::VulkanBuffer(buffer) => Device::VulkanDevice(buffer.device().clone()),
        }
    }
    pub fn vulkan(&self) -> Option<&VulkanBuffer> {
        match self {
            Self::VulkanBuffer(buffer) => Some(buffer),
            _ => None,
        }
    }
    pub fn cuda(&self) -> Option<&CudaBuffer> {
        match self {
            Self::CudaBuffer(buffer) => Some(buffer),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum Texture {
    VulkanTexture(VulkanTexture),
}

impl Texture {
    pub fn vulkan(&self) -> Option<&VulkanTexture> {
        match self {
            Self::VulkanTexture(buffer) => Some(buffer),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum Accel {
    VulkanAccel(VulkanAccel),
}
impl Accel {
    pub fn vulkan(&self) -> Option<&VulkanAccel> {
        match self {
            Self::VulkanAccel(accel) => Some(accel),
            _ => None,
        }
    }
}

pub trait BackendDevice: Clone {
    type Buffer: BackendBuffer;
    type Texture: BackendTexture;
    type Accel: BackendAccel;
    fn create_buffer(&self, size: usize) -> Result<Self::Buffer>;
    fn create_buffer_from_slice(&self, slice: &[u8]) -> Result<Self::Buffer>;
    fn create_texture(&self, shape: [usize; 3], channels: usize) -> Result<Self::Texture>;
    fn create_accel(&self, desc: &AccelDesc) -> Result<Self::Accel>;
    fn execute_ir(&self, ir: &IR, num: usize, buffers: &[&Self::Buffer]) -> Result<()>;
    fn execute_graph(&self, trace: &trace::Trace, graph: &Graph) -> Result<()> {
        todo!()
    }
}

pub trait BackendBuffer {
    type Device: BackendDevice;
    fn to_host<T: bytemuck::Pod>(&self) -> Result<Vec<T>>;
    fn size(&self) -> usize;
    fn device(&self) -> &Self::Device;
}

impl AsRef<CudaBuffer> for Buffer {
    fn as_ref(&self) -> &CudaBuffer {
        match self {
            Buffer::CudaBuffer(buffer) => buffer,
            _ => todo!(),
        }
    }
}

pub trait BackendTexture: Debug {
    type Device: BackendDevice;
}

pub trait BackendAccel {
    type Device: BackendDevice;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryDesc {
    Triangles {
        n_triangles: usize,
        n_vertices: usize,
    },
}
// #[derive(Debug, Clone, PartialEq)]
// pub struct InstanceDesc {
//     pub geometry: usize,
//     pub transform: [f32; 12],
// }
/// TODO: At some point we should probably rename this to AccelExtent and have a unified instance
/// descriptor on the device.
/// This would allow us to change instance transforms on the fly (necessary for differentiable
/// rendering)
#[derive(Debug, Clone, PartialEq)]
pub struct AccelDesc {
    pub geometries: Vec<GeometryDesc>,
    pub instances: usize,
}
