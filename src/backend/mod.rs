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
    pub fn create_buffer(&self, size: usize) -> Result<Buffer> {
        let buffer = match self {
            Device::CudaDevice(device) => Buffer::CudaBuffer(Arc::new(device.create_buffer(size)?)),
            Device::VulkanDevice(device) => {
                Buffer::VulkanBuffer(Arc::new(device.create_buffer(size)?))
            }
        };
        Ok(buffer)
    }
    pub fn execute_ir(&self, ir: &IR, num: usize, buffers: &[Buffer]) -> Result<()> {
        match self {
            Device::CudaDevice(device) => {
                let buffers = buffers
                    .iter()
                    .map(|b| match b {
                        Buffer::CudaBuffer(buffer) => buffer.as_ref(),
                        _ => todo!(),
                    })
                    .collect::<Vec<_>>();
                device.execute_ir(ir, num, &buffers)
            }
            Device::VulkanDevice(device) => {
                let buffers = buffers
                    .iter()
                    .map(|b| match b {
                        Buffer::VulkanBuffer(buffer) => buffer.as_ref(),
                        _ => todo!(),
                    })
                    .collect::<Vec<_>>();
                device.execute_ir(ir, num, &buffers)
            }
        }
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
    pub fn device(&self) -> Device {
        match self {
            Buffer::CudaBuffer(buffer) => Device::CudaDevice(buffer.device().clone()),
            Buffer::VulkanBuffer(buffer) => Device::VulkanDevice(buffer.device().clone()),
        }
    }
}

pub trait BackendDevice: Clone {
    type Buffer: BackendBuffer;
    fn create_buffer(&self, size: usize) -> Result<Self::Buffer>;
    fn execute_ir(&self, ir: &IR, num: usize, buffers: &[&Self::Buffer]) -> Result<()>;
}

pub trait BackendBuffer {
    type Device: BackendDevice;
    fn to_host<T: bytemuck::Pod>(&self) -> Result<Vec<T>>;
    fn size(&self) -> usize;
    fn device(&self) -> &Self::Device;
}

pub trait Texture {
    type Device: BackendDevice;
}
