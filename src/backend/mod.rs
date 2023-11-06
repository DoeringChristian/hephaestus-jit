mod cuda;
mod vulkan;
use std::ops::Deref;
use std::sync::Arc;

use cuda::{CudaBuffer, CudaDevice};
use vulkan::{VulkanBuffer, VulkanDevice};

use crate::ir::IR;
use crate::vartype::AsVarType;

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
    pub fn create_buffer<T: AsVarType>(&self, len: usize) -> Result<Buffer> {
        let ty = T::var_ty();
        let size = len * ty.size();
        let buffer = match self {
            Device::CudaDevice(device) => Buffer::CudaBuffer(Arc::new(device.create_buffer(size)?)),
            Device::VulkanDevice(device) => {
                Buffer::VulkanBuffer(Arc::new(device.create_buffer(size)?))
            }
        };
        Ok(buffer)
    }
}

#[derive(Debug, Clone)]
pub enum Buffer {
    CudaBuffer(Arc<CudaBuffer>),
    VulkanBuffer(Arc<VulkanBuffer>),
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
}

pub trait BackendDevice: Clone {
    type Buffer: BackendBuffer;
    fn create_buffer(&self, size: usize) -> Result<Self::Buffer>;
    fn execute_trace(&self, trace: &IR, arrays: &[&Self::Buffer]) -> Result<()>;
}

pub trait BackendBuffer {
    type Device: BackendDevice;
    fn to_host<T: bytemuck::Pod>(&self) -> Result<Vec<T>>;
    fn size(&self) -> usize;
}

pub trait Texture {
    type Device: BackendDevice;
}
