mod cuda;
mod report;
mod vulkan;

use std::fmt::Debug;
use std::sync::Arc;

use cuda::{CudaBuffer, CudaDevice};
use vulkan::{VulkanBuffer, VulkanDevice, VulkanTexture};

use crate::graph::Env;
use crate::graph::Graph;
use crate::vartype::AsVarType;
use crate::vartype::VarType;

pub use report::{ExecReport, PassReport, Report};

use self::vulkan::VulkanAccel;

// Backend Traits:

///
/// This trait represents an interface to a device.
/// It has to be implemented for the device.
///
pub trait BackendDevice: Clone + Send + Sync {
    type Buffer: BackendBuffer;
    type Texture: BackendTexture;
    type Accel: BackendAccel;
    fn create_buffer(&self, size: usize) -> Result<Self::Buffer>;
    fn create_buffer_from_slice(&self, slice: &[u8]) -> Result<Self::Buffer>;
    fn create_texture(&self, desc: &TextureDesc) -> Result<Self::Texture>;
    fn create_accel(&self, desc: &AccelDesc) -> Result<Self::Accel>;
    fn execute_graph(&self, graph: &Graph, env: &crate::graph::Env) -> Result<Report>;
}

pub trait BackendBuffer: Clone + Send + Sync {
    type Device: BackendDevice;
    fn to_host<T: AsVarType>(&self, range: std::ops::Range<usize>) -> Result<Vec<T>>;
    fn device(&self) -> &Self::Device;
}

pub trait BackendTexture: Debug + Clone + Send + Sync {
    type Device: BackendDevice;
}

pub trait BackendAccel: Clone + Send + Sync {
    type Device: BackendDevice;
}

// Device:

// TODO: Device Caching
pub fn vulkan(id: usize) -> Device {
    Device::vulkan(id).unwrap()
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    VulkanError(#[from] vulkan::Error),
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
    pub fn as_vulkan(&self) -> Option<&VulkanDevice> {
        match self {
            Device::VulkanDevice(device) => Some(device),
            _ => None,
        }
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
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture> {
        match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => {
                Ok(Texture::VulkanTexture(device.create_texture(desc)?))
            }
        }
    }
    pub fn create_accel(&self, desc: &AccelDesc) -> Result<Accel> {
        match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => Ok(Accel::VulkanAccel(device.create_accel(desc)?)),
        }
    }
    pub fn execute_graph(&self, graph: &Graph, env: &Env) -> Result<Report> {
        let report = match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => device.execute_graph(graph, env),
        };
        if let Ok(report) = &report {
            report.submit_to_profiler();
        };
        report
    }
}

// Buffer

// TODO: we might not need it to be cloneable
#[derive(Debug, Clone)]
pub enum Buffer {
    CudaBuffer(CudaBuffer),
    VulkanBuffer(VulkanBuffer),
}
impl Buffer {
    pub fn to_host<T: AsVarType>(&self, range: std::ops::Range<usize>) -> Result<Vec<T>> {
        match self {
            Self::CudaBuffer(buffer) => Ok(buffer.to_host(range)?),
            Self::VulkanBuffer(buffer) => Ok(buffer.to_host(range)?),
        }
    }
    pub fn device(&self) -> Device {
        match self {
            Buffer::CudaBuffer(buffer) => Device::CudaDevice(buffer.device().clone()),
            Buffer::VulkanBuffer(buffer) => Device::VulkanDevice(buffer.device().clone()),
        }
    }
    pub fn cuda(&self) -> Option<&CudaBuffer> {
        match self {
            Self::CudaBuffer(buffer) => Some(buffer),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Texture {
    VulkanTexture(VulkanTexture),
}

impl Texture {}

#[derive(Debug, Clone)]
pub enum Accel {
    VulkanAccel(VulkanAccel),
}
impl Accel {}

impl AsRef<CudaBuffer> for Buffer {
    fn as_ref(&self) -> &CudaBuffer {
        match self {
            Buffer::CudaBuffer(buffer) => buffer,
            _ => todo!(),
        }
    }
}

// Descriptors:

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size: usize,
    pub ty: &'static VarType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureDesc {
    pub shape: [usize; 3],
    pub channels: usize,
    pub format: &'static VarType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeometryDesc {
    Triangles {
        // TODO: add stride
        n_triangles: usize,
        n_vertices: usize,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AccelDesc {
    pub geometries: Arc<[GeometryDesc]>,
    pub instances: usize,
}
