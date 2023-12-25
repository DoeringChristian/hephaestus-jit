mod codegen;
mod param_layout;

// pub use cudarc::driver::DriverError;
// use cudarc::driver::{self as core, sys, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::backend::{self, AccelDesc, BackendBuffer, BackendDevice};
use crate::vartype::AsVarType;

#[derive(Clone, Debug)]
pub struct CudaDevice {
    // device: Arc<core::CudaDevice>,
}

impl CudaDevice {
    pub fn create(id: usize) -> backend::Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct CudaBuffer {
    device: CudaDevice,
    // buffer: core::CudaSlice<u8>,
    size: usize,
}

impl BackendDevice for CudaDevice {
    type Texture = CudaTexture;
    type Buffer = CudaBuffer;
    type Accel = CudaAccel;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        todo!()
    }

    fn execute_ir(
        &self,
        ir: &crate::ir::IR,
        num: usize,
        buffers: &[&Self::Buffer],
    ) -> backend::Result<()> {
        todo!()
    }

    fn create_texture(&self, shape: [usize; 3], channels: usize) -> backend::Result<Self::Texture> {
        todo!()
    }

    fn create_buffer_from_slice(&self, slice: &[u8]) -> backend::Result<Self::Buffer> {
        todo!()
    }

    fn create_accel(&self, desc: &AccelDesc) -> backend::Result<Self::Accel> {
        todo!()
    }
}

impl BackendBuffer for CudaBuffer {
    type Device = CudaDevice;

    fn to_host<T: AsVarType>(&self, range: std::ops::Range<usize>) -> backend::Result<Vec<T>> {
        todo!()
    }

    fn size(&self) -> usize {
        todo!()
    }

    fn device(&self) -> &Self::Device {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct CudaTexture;

impl backend::BackendTexture for CudaTexture {
    type Device = CudaDevice;
}

#[derive(Debug, Clone)]
pub struct CudaAccel;

impl backend::BackendAccel for CudaAccel {
    type Device = CudaDevice;
}
