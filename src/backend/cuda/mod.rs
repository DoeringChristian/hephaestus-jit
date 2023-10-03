mod core;
use crate::backend::{BackendBuffer, BackendDevice};

#[derive(Clone, Debug)]
pub struct CudaDevice(core::Device);

pub struct CudaBuffer(core::Lease<core::Buffer>);

impl BackendDevice for CudaDevice {
    type Buffer = CudaBuffer;
}

impl BackendBuffer for CudaBuffer {
    type Device = CudaDevice;
}
