mod cuda;
use std::fmt::Debug;
use std::sync::Arc;

use cuda::CudaDevice;

use crate::trace::Trace;

use self::cuda::CudaArray;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("[CudaError] {0:?}")]
    CudaError(#[from] cuda::DriverError),
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
pub enum Device {
    CudaDevice(CudaDevice),
}

impl Device {
    pub fn cuda(id: usize) -> Result<Self> {
        Ok(Device::CudaDevice(CudaDevice::create(id)?))
    }
    pub fn create_array<T: bytemuck::Pod>(&self, size: usize) -> Result<Array<T>> {
        match self {
            Self::CudaDevice(device) => Ok(Array::CudaArray(Arc::new(device.create_array(size)?))),
        }
    }
    pub fn execute_trace(&self, trace: &Trace, params: Parameters) -> Result<()> {
        match self {
            Self::CudaDevice(device) => device.execute_trace(trace, params),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Array<T> {
    CudaArray(Arc<CudaArray<T>>),
}

impl<T: bytemuck::Pod> Array<T> {
    pub fn to_host(&self) -> Result<Vec<T>> {
        match self {
            Array::CudaArray(array) => array.to_host(),
        }
    }
}

pub trait BackendDevice: Clone {
    type Array<T: bytemuck::Pod>: BackendArray<T>;
    fn create_array<T: bytemuck::Pod>(&self, size: usize) -> Result<Self::Array<T>>;
    fn execute_trace(&self, trace: &Trace, params: Parameters) -> Result<()>;
}

pub trait BackendArray<T> {
    type Device: BackendDevice;
    fn to_host(&self) -> Result<Vec<T>>;
}

pub trait Parameter: Debug {}

impl<T: Debug> Parameter for Array<T> {}

#[derive(Debug)]
pub struct Parameters {
    pub size: u32,
    pub arrays: Vec<Box<dyn Parameter>>,
}
