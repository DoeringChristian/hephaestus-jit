mod cuda;
use cuda::CudaDevice;

#[derive(Clone)]
pub enum Device {
    CudaDevice(CudaDevice),
}

pub trait BackendDevice: Clone {
    type Buffer: BackendBuffer;
}

pub trait BackendBuffer {
    type Device: BackendDevice;
}
