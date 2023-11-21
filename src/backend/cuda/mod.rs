mod codegen;
mod cuda_core;
mod param_layout;

pub use cudarc::driver::DriverError;
use cudarc::driver::{self as core, sys, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::backend::{self, AccelDesc, BackendBuffer, BackendDevice};

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
    type Texture = CudaTexture;
    type Buffer = CudaBuffer;
    type Accel = CudaAccel;

    fn create_buffer(&self, size: usize) -> backend::Result<Self::Buffer> {
        Ok(CudaBuffer {
            device: self.clone(),
            buffer: unsafe { self.device.alloc(size)? },
            size,
        })
    }

    fn execute_ir(
        &self,
        ir: &crate::ir::IR,
        num: usize,
        buffers: &[&Self::Buffer],
    ) -> backend::Result<()> {
        let mut asm = String::new();
        codegen::assemble_trace(&mut asm, &ir, "main", "global").unwrap();

        print!("{asm}");
        std::fs::write("/tmp/cuda.ptx", &asm).unwrap();
        self.device
            .load_ptx(cudarc::nvrtc::Ptx::from_src(asm), "kernels", &["main"])
            .unwrap();

        let func = self.device.get_func("kernels", "main").unwrap();

        let param_buffer = buffers
            .iter()
            .map(|a| *a.buffer.device_ptr())
            .collect::<Vec<_>>();
        let param_buffer = self.device.htod_sync_copy(&param_buffer).unwrap();

        println!("{:#x?}", param_buffer.device_ptr());

        let cfg = LaunchConfig::for_num_elems(num as _);
        dbg!(&cfg);

        unsafe {
            func.launch(cfg, (num as u64, *(param_buffer.device_ptr())))
                .unwrap()
        };

        self.device.synchronize().unwrap();
        drop(param_buffer);

        Ok(())
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

    fn to_host<T: bytemuck::Pod>(&self) -> backend::Result<Vec<T>> {
        let len = self.size() / std::mem::size_of::<T>();
        let mut dst = Vec::<T>::with_capacity(len);
        unsafe { dst.set_len(len) };
        self.device
            .device
            .dtoh_sync_copy_into(&self.buffer, bytemuck::cast_slice_mut::<_, u8>(&mut dst))?;
        Ok(dst)
    }

    fn size(&self) -> usize {
        self.size
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }
}

#[derive(Debug)]
pub struct CudaTexture;

impl backend::BackendTexture for CudaTexture {
    type Device = CudaDevice;
}

#[derive(Debug)]
pub struct CudaAccel;

impl backend::BackendAccel for CudaAccel {
    type Device = CudaDevice;
}
