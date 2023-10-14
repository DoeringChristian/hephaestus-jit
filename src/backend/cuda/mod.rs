mod codegen;
mod param_layout;

pub use cudarc::driver::DriverError;
use cudarc::driver::{self as core, sys, DevicePtr, LaunchAsync, LaunchConfig};
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
    device: CudaDevice,
    buffer: core::CudaSlice<u8>,
    size: usize,
}

impl BackendDevice for CudaDevice {
    type Buffer = CudaBuffer;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        Ok(CudaBuffer {
            device: self.clone(),
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
        codegen::assemble_trace(&mut asm, &trace, "main", "global").unwrap();

        print!("{asm}");
        std::fs::write("/tmp/cuda.ptx", &asm).unwrap();
        self.device
            .load_ptx(cudarc::nvrtc::Ptx::from_src(asm), "kernels", &["main"])
            .unwrap();

        let func = self.device.get_func("kernels", "main").unwrap();

        let param_buffer = params_buffer(&params);
        println!("{:#x?}", param_buffer);
        let param_buffer = self.device.htod_sync_copy(&param_buffer).unwrap();

        println!("{:#x?}", param_buffer.device_ptr());

        let cfg = LaunchConfig::for_num_elems(params.size);

        unsafe {
            func.launch(cfg, (params.size as u64, *(param_buffer.device_ptr())))
                .unwrap()
        };

        self.device.synchronize().unwrap();
        drop(param_buffer);

        Ok(())
    }
}

impl BackendBuffer for CudaBuffer {
    type Device = CudaDevice;

    fn to_host(&self) -> backend::Result<Vec<u8>> {
        Ok(self.device.device.dtoh_sync_copy(&self.buffer)?)
    }
}

pub fn params_buffer(params: &backend::Parameters) -> Vec<u64> {
    let buffers = params.buffers.iter().map(|b| *match b.as_ref() {
        backend::Buffer::CudaBuffer(buffer) => {
            println!("{:#x?}", buffer.buffer.device_ptr());
            buffer.buffer.device_ptr()
        }
        _ => todo!(),
    });

    buffers.collect()
}
