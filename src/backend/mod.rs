mod cuda;
mod vulkan;
use std::ops::Deref;
use std::sync::Arc;

use cuda::{CudaBuffer, CudaDevice};
use vulkan::{VulkanBuffer, VulkanDevice};

use crate::ir::IR;
use crate::vartype::{AsVarType, VarType};

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
    pub fn create_array<T: AsVarType>(&self, len: usize) -> Result<Array> {
        let ty = T::var_ty();
        let size = len * ty.size();
        let buffer = match self {
            Device::CudaDevice(device) => Buffer::CudaBuffer(device.create_buffer(size)?),
            Device::VulkanDevice(device) => Buffer::VulkanBuffer(device.create_buffer(size)?),
        };
        Ok(Array(Arc::new(InternalArray { ty, buffer })))
    }
    pub fn execute_trace(&self, trace: &IR, arrays: &[Array]) -> Result<()> {
        match self {
            Self::CudaDevice(device) => {
                let buffers = arrays
                    .iter()
                    .map(|a| match a.buffer() {
                        Buffer::CudaBuffer(buffer) => buffer,
                        _ => todo!(),
                    })
                    .collect::<Vec<_>>();
                device.execute_trace(trace, &buffers)
            }
            Self::VulkanDevice(device) => {
                let buffers = arrays
                    .iter()
                    .map(|a| match a.buffer() {
                        Buffer::VulkanBuffer(buffer) => buffer,
                        _ => todo!(),
                    })
                    .collect::<Vec<_>>();
                device.execute_trace(trace, &buffers)
            }
        }
    }
}

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
}
#[derive(Debug)]
struct InternalArray {
    ty: VarType,
    buffer: Buffer,
}
#[derive(Debug, Clone)]
pub struct Array(Arc<InternalArray>);

// impl Deref for Array {
//     type Target = Buffer;
//
//     fn deref(&self) -> &Self::Target {
//         &self.0.buffer
//     }
// }

impl Array {
    pub fn ty(&self) -> VarType {
        self.0.ty.clone()
    }
    pub fn size(&self) -> usize {
        self.0.buffer.size()
    }
    pub fn len(&self) -> usize {
        self.size() / self.ty().size()
    }
    fn buffer(&self) -> &Buffer {
        &self.0.buffer
    }
    pub fn to_host<T: bytemuck::Pod + AsVarType>(&self) -> Result<Vec<T>> {
        assert_eq!(T::var_ty(), self.ty(), "Types do not match!");
        self.0.buffer.to_host()
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
