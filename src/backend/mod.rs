mod cuda;
mod vulkan;
use std::sync::Arc;

use cuda::{CudaArray, CudaDevice};
use vulkan::{VulkanArray, VulkanDevice};

use crate::trace::{Trace, VarType};

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
    pub fn create_array(&self, size: usize, ty: VarType) -> Result<Array> {
        match self {
            Self::CudaDevice(device) => Ok(Array::CudaArray(
                Arc::new(device.create_array(size * ty.size())?),
                ty,
            )),
            Self::VulkanDevice(device) => Ok(Array::VulkanArray(
                Arc::new(device.create_array(size * ty.size())?),
                ty,
            )),
        }
    }
    pub fn execute_trace(&self, trace: &Trace, params: Parameters) -> Result<()> {
        match self {
            Self::CudaDevice(device) => device.execute_trace(trace, params),
            Self::VulkanDevice(device) => device.execute_trace(trace, params),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Array {
    CudaArray(Arc<CudaArray>, VarType),
    VulkanArray(Arc<VulkanArray>, VarType),
}

impl Array {
    pub fn to_host<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
        match self {
            Array::CudaArray(array, _) => Ok(bytemuck::cast_vec(array.to_host()?)),
            Array::VulkanArray(array, _) => Ok(bytemuck::cast_vec(array.to_host()?)),
        }
    }
    pub fn ty(&self) -> VarType {
        match self {
            Array::CudaArray(_, ty) => ty.clone(),
            Array::VulkanArray(_, ty) => ty.clone(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Array::CudaArray(array, ty) => array.size() / ty.size(),
            Array::VulkanArray(array, ty) => array.size() / ty.size(),
        }
    }
}

pub trait BackendDevice: Clone {
    type Array: BackendArray;
    fn create_array(&self, size: usize) -> Result<Self::Array>;
    fn execute_trace(&self, trace: &Trace, params: Parameters) -> Result<()>;
}

pub trait BackendArray {
    type Device: BackendDevice;
    fn to_host(&self) -> Result<Vec<u8>>;
    fn size(&self) -> usize;
}

#[derive(Debug)]
pub struct Parameters {
    pub size: u32,
    pub arrays: Vec<Array>,
}
