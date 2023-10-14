mod cuda;
use std::sync::Arc;

use cuda::CudaDevice;

use crate::trace::Trace;

use self::cuda::CudaBuffer;

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
    pub fn create_buffer(&self, size: usize) -> Result<Buffer> {
        match self {
            Self::CudaDevice(device) => Ok(Buffer::CudaBuffer(device.create_buffer(size)?)),
        }
    }
    pub fn execute_trace(&self, trace: &Trace, params: Parameters) -> Result<()> {
        match self {
            Self::CudaDevice(device) => device.execute_trace(trace, params),
        }
    }
}

#[derive(Debug)]
pub enum Buffer {
    CudaBuffer(CudaBuffer),
}

impl Buffer {}

pub trait BackendDevice: Clone {
    type Buffer: BackendBuffer;
    fn create_buffer(&self, size: usize) -> Result<Self::Buffer>;
    fn execute_trace(&self, trace: &Trace, params: Parameters) -> Result<()>;
}

pub trait BackendBuffer {
    type Device: BackendDevice;
}

#[derive(Debug)]
pub struct Parameters {
    pub buffers: Vec<Arc<Buffer>>,
}
