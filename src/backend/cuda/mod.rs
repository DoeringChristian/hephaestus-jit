mod codegen;
mod param_layout;

use cudarc::driver as core;
pub use cudarc::driver::DriverError;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

#[cfg(test)]
mod test;

use crate::backend::{self, BackendBuffer, BackendDevice};

#[derive(Clone, Debug)]
pub struct CudaDevice {
    device: Arc<core::CudaDevice>,
}

impl CudaDevice {
    pub fn create(id: usize) -> backend::Result<Self> {
        Ok(Self {
            device: core::CudaDevice::new(id)?,
        })
    }
}

#[derive(Debug)]
pub struct CudaBuffer {
    buffer: core::CudaSlice<u8>,
    size: usize,
}

impl BackendDevice for CudaDevice {
    type Buffer = CudaBuffer;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        Ok(CudaBuffer {
            buffer: unsafe { self.device.alloc(size)? },
            size,
        })
    }

    fn execute_trace(
        &self,
        trace: &crate::trace::Trace,
        params: backend::Parameters,
    ) -> backend::Result<()> {
        let mut asm = String::new();
        codegen::assemble_trace(&mut asm, &trace, "main", "param").unwrap();

        print!("{asm}");
        std::fs::write("/tmp/cuda.ptx", &asm).unwrap();
        self.device
            .load_ptx(Ptx::from_src(asm), "test", &["main"])
            .unwrap();
        todo!()
    }
}

impl BackendBuffer for CudaBuffer {
    type Device = CudaDevice;
}
