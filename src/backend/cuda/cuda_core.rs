use std::ffi::{c_void, CStr, CString};
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use parking_lot::Mutex;
use resource_pool::hashpool::*;

use cuda_rs::{
    CUcontext, CUdevice_attribute, CUevent, CUevent_flags, CUstream, CudaApi, CudaError,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CudaError(#[from] CudaError),
    #[error("Loading Error {}", .0)]
    Loading(#[from] libloading::Error),
    #[error("The CUDA verion {}.{} is not supported!", .0, .1)]
    VersionError(i32, i32),
}

pub struct CtxRef {
    device: Arc<InternalDevice>,
}

impl CtxRef {
    pub fn create(device: &Arc<InternalDevice>) -> Self {
        unsafe { device.instance.api.cuCtxPushCurrent_v2(device.ctx) };
        Self {
            device: device.clone(),
        }
    }
    pub fn raw(&self) -> CUcontext {
        self.device.ctx
    }
}
impl Deref for CtxRef {
    type Target = CudaApi;

    fn deref(&self) -> &Self::Target {
        &self.device.instance.api
    }
}
impl Drop for CtxRef {
    fn drop(&mut self) {
        unsafe {
            let mut old_ctx = std::ptr::null_mut();
            self.device.instance.api.cuCtxPopCurrent_v2(&mut old_ctx);
        }
    }
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub pci_bus_id: i32,
    pub pci_dom_id: i32,
    pub pci_dev_id: i32,
    pub num_sm: i32,
    pub unified_addr: i32,
    pub shared_memory_bytes: i32,
    pub cc_minor: i32,
    pub cc_major: i32,
    pub memory_pool: i32,
    pub mem_total: usize,
    pub name: String,
}

impl Debug for InternalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDADevice({})", self.info.name)
    }
}
pub struct InternalDevice {
    id: i32,
    ctx: CUcontext,
    instance: Arc<Instance>,
    info: DeviceInfo,
}

impl InternalDevice {
    pub fn new(instance: &Arc<Instance>, id: i32) -> Result<Self, Error> {
        unsafe {
            let api = &instance.api;
            let mut ctx = std::ptr::null_mut();
            api.cuDevicePrimaryCtxRetain(&mut ctx, id).check()?;
            api.cuCtxPushCurrent_v2(ctx).check()?;

            let mut pci_bus_id = 0;
            let mut pci_dom_id = 0;
            let mut pci_dev_id = 0;
            let mut num_sm = 0;
            let mut unified_addr = 0;
            let mut shared_memory_bytes = 0;
            let mut cc_minor = 0;
            let mut cc_major = 0;
            let mut memory_pool = 0;

            let mut mem_total = 0;

            let mut name = [0u8; 256];

            api.cuDeviceGetName(name.as_mut_ptr() as *mut _, name.len() as _, id)
                .check()?;
            api.cuDeviceGetAttribute(
                &mut pci_bus_id,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut pci_dev_id,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut pci_dom_id,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut num_sm,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut unified_addr,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut shared_memory_bytes,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut cc_minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut cc_major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                id,
            )
            .check()?;
            api.cuDeviceTotalMem_v2(&mut mem_total, id).check()?;

            if instance.cuda_version_major > 11
                || (instance.cuda_version_major == 11 && instance.cuda_version_minor >= 2)
            {
                api.cuDeviceGetAttribute(
                    &mut memory_pool,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                    id,
                )
                .check()?;
            }

            let name = CStr::from_bytes_until_nul(&name).unwrap();
            let name = String::from_utf8_lossy(name.to_bytes()).to_string();

            log::trace!(
                "Found CUDA device {id}: \"{name}\" (PCI ID\
                    {pci_bus_id:#04x}:{pci_dev_id:#04x}.{pci_dom_id}, compute cap.\
                    {cc_major}.{cc_minor}, {num_sm} SMs w/{shared_memory_bytes} shared mem., \
                    {mem_total} global mem.)",
                shared_memory_bytes = bytesize::ByteSize(shared_memory_bytes as _),
                mem_total = bytesize::ByteSize(mem_total as _)
            );

            let mut old_ctx = std::ptr::null_mut();
            api.cuCtxPopCurrent_v2(&mut old_ctx).check()?;

            Ok(Self {
                ctx,
                instance: instance.clone(),
                id,
                info: DeviceInfo {
                    pci_bus_id,
                    pci_dom_id,
                    pci_dev_id,
                    num_sm,
                    unified_addr,
                    shared_memory_bytes,
                    cc_minor,
                    cc_major,
                    memory_pool,
                    mem_total,
                    name,
                },
            })
        }
    }
    pub fn ctx(self: &Arc<InternalDevice>) -> CtxRef {
        CtxRef::create(self)
    }
}
impl Drop for InternalDevice {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .api
                .cuDevicePrimaryCtxRelease_v2(self.id)
                .check()
                .unwrap()
        }
    }
}

impl InternalDevice {
    pub fn t(self: &Arc<InternalDevice>) {}
}

#[derive(Clone)]
pub struct Device {
    internal: Arc<InternalDevice>,
    buffer_pool: Arc<Mutex<HashPool<BufferDesc>>>,
}
impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDADevice({})", self.info().name)
    }
}
unsafe impl Sync for Device {}
unsafe impl Send for Device {}
// impl Debug for Device {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{:?}", self.internal)
//     }
// }

impl Device {
    pub fn create(instance: &Arc<Instance>, id: i32) -> Result<Self, Error> {
        let internal = Arc::new(InternalDevice::new(&instance, id)?);

        Ok(Self {
            internal,
            buffer_pool: Arc::new(Mutex::new(Default::default())),
        })
    }
    // TODO: better context switch operation
    pub fn ctx(&self) -> CtxRef {
        CtxRef::create(&self.internal)
    }
    pub fn create_stream(&self, flags: cuda_rs::CUstream_flags_enum) -> Result<Stream, Error> {
        Stream::create(self, flags)
    }
    pub fn info(&self) -> &DeviceInfo {
        &self.internal.info
    }
    pub fn buffer_uninit(&self, size: usize) -> Result<Buffer, Error> {
        Buffer::uninit(self, size)
    }
    pub fn buffer_from_slice(&self, slice: &[u8]) -> Result<Buffer, Error> {
        let buf = Buffer::uninit(self, slice.len())?;
        buf.copy_from_slice(slice)?;
        Ok(buf)
    }
    pub fn lease_buffer(&self, size: usize) -> Result<Lease<Buffer>, Error> {
        let size = round_pow2(size as _) as usize;
        self.buffer_pool
            .lock()
            .try_lease(&BufferDesc { size }, self)
    }
    pub fn create_texture(&self, desc: &TexutreDesc) -> Result<Texture, Error> {
        Texture::create(self, desc)
    }
}

pub struct Instance {
    api: CudaApi,
    device_count: i32,
    cuda_version_major: i32,
    cuda_version_minor: i32,
}
unsafe impl Sync for Instance {}
unsafe impl Send for Instance {}
impl Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            // .field("api", &self.api)
            .field("device_count", &self.device_count)
            .field("cuda_version_major", &self.cuda_version_major)
            .field("cuda_version_minor", &self.cuda_version_minor)
            .finish()
    }
}
impl Instance {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            let api = CudaApi::find_and_load()?;
            api.cuInit(0).check()?;

            let mut device_count = 0;
            api.cuDeviceGetCount(&mut device_count).check()?;

            let mut cuda_version = 0;
            api.cuDriverGetVersion(&mut cuda_version).check()?;

            let cuda_version_major = cuda_version / 1000;
            let cuda_version_minor = (cuda_version % 1000) / 10;

            log::trace!(
                "Found CUDA driver with version {}.{}",
                cuda_version_major,
                cuda_version_minor
            );

            if cuda_version_major < 10 {
                log::error!(
                    "CUDA version {}.{} is to old an not supported. The minimum supported\
                    version is 10.x",
                    cuda_version_major,
                    cuda_version_minor
                );
                return Err(Error::VersionError(cuda_version_major, cuda_version_minor));
            }
            Ok(Self {
                api,
                device_count,
                cuda_version_major,
                cuda_version_minor,
            })
        }
    }
    pub fn device_count(&self) -> usize {
        self.device_count as _
    }
}

impl Drop for Instance {
    fn drop(&mut self) {}
}

#[derive(Debug)]
pub struct Stream {
    events: Vec<Arc<Event>>,
    raw: CUstream,
    device: Arc<InternalDevice>,
}

impl Stream {
    pub fn create(device: &Device, flags: cuda_rs::CUstream_flags_enum) -> Result<Self, Error> {
        let ctx = device.ctx();
        unsafe {
            let mut stream = std::ptr::null_mut();
            ctx.cuStreamCreate(&mut stream, flags as _).check()?;

            Ok(Self {
                raw: stream,
                device: device.internal.clone(),
                events: vec![],
            })
        }
    }
    #[must_use]
    pub fn synchronize(&self) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuStreamSynchronize(self.raw).check()?;
        }
        Ok(())
    }
    pub unsafe fn raw(&self) -> CUstream {
        self.raw
    }
    pub fn record_event(&mut self, event: &Arc<Event>) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuEventRecord(event.raw(), self.raw()).check()?;
        }
        self.events.push(event.clone());
        Ok(())
    }
}

impl Deref for Stream {
    type Target = CUstream;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuStreamSynchronize(self.raw).check().unwrap();
            ctx.cuStreamDestroy_v2(self.raw).check().unwrap();
        }
    }
}

#[derive(Debug)]
pub struct InternalModule {
    module: cuda_rs::CUmodule,
    device: Arc<InternalDevice>,
}

impl Drop for InternalModule {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuModuleUnload(self.module);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Module(Arc<InternalModule>);

impl Module {
    pub fn from_ptx(device: &Device, ptx: &str) -> Result<Module, Error> {
        let ctx = device.ctx();
        unsafe {
            let ptx_cstring = CString::new(ptx).unwrap();

            const LOG_SIZE: usize = 16384;
            let mut error_log = [0u8; LOG_SIZE];
            let mut info_log = [0u8; LOG_SIZE];

            let mut options = [
                cuda_rs::CUjit_option_enum::CU_JIT_OPTIMIZATION_LEVEL,
                cuda_rs::CUjit_option_enum::CU_JIT_LOG_VERBOSE,
                cuda_rs::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER,
                cuda_rs::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                cuda_rs::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER,
                cuda_rs::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                cuda_rs::CUjit_option_enum::CU_JIT_GENERATE_LINE_INFO,
                cuda_rs::CUjit_option_enum::CU_JIT_GENERATE_DEBUG_INFO,
            ];

            let mut option_values = [
                4 as *mut c_void,
                1 as *mut c_void,
                info_log.as_mut_ptr() as *mut c_void,
                LOG_SIZE as *mut c_void,
                error_log.as_mut_ptr() as *mut c_void,
                LOG_SIZE as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
            ];

            let mut linkstate = std::ptr::null_mut();
            ctx.cuLinkCreate_v2(
                options.len() as _,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
                &mut linkstate,
            )
            .check()?;

            ctx.cuLinkAddData_v2(
                linkstate,
                cuda_rs::CUjitInputType::CU_JIT_INPUT_PTX,
                ptx_cstring.as_ptr() as *mut c_void,
                ptx.len(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .check()
            .or_else(|err| {
                let error_log = CStr::from_bytes_until_nul(&error_log).unwrap().to_str().unwrap();
                log::error!("Compilation failed. Please see the PTX listing and error message below:\n{}\n{}", error_log, err);
                Err(err)
            })?;

            let mut link_out = std::ptr::null_mut();
            let mut link_out_size = 0;
            ctx.cuLinkComplete(linkstate, &mut link_out, &mut link_out_size)
                .check()
                .or_else(|err| {
                    let error_log = CStr::from_bytes_until_nul(&error_log).unwrap().to_str().unwrap();
                    log::error!("Compilation failed. Please see the PTX listing and error message below:\n{}\n{}", error_log, err);
                    Err(err)
                })?;

            log::trace!(
                "Detailed linker output: {}",
                CStr::from_bytes_until_nul(&info_log)
                    .unwrap()
                    .to_str()
                    .unwrap()
            );

            let mut out: Vec<u8> = Vec::with_capacity(link_out_size);
            std::ptr::copy_nonoverlapping(link_out as *mut u8, out.as_mut_ptr(), link_out_size);
            out.set_len(link_out_size);

            ctx.cuLinkDestroy(linkstate).check()?;

            let mut module = std::ptr::null_mut();
            ctx.cuModuleLoadData(&mut module, out.as_ptr() as *const c_void)
                .check()?;

            Ok(Self(Arc::new(InternalModule {
                module,
                device: device.internal.clone(),
            })))
        }
    }
    pub fn function(&self, name: &str) -> Result<Function, Error> {
        let ctx = self.0.device.ctx();
        unsafe {
            let fname = CString::new(name).unwrap();
            let mut func = std::ptr::null_mut();
            ctx.cuModuleGetFunction(&mut func, self.0.module, fname.as_ptr() as *const i8)
                .check()?;
            Ok(Function {
                func,
                module: self.clone(),
            })
        }
    }
}

#[derive(Debug)]
pub struct Function {
    func: cuda_rs::CUfunction,
    module: Module,
}

impl Function {
    pub unsafe fn raw(&self) -> cuda_rs::CUfunction {
        self.func
    }
    pub unsafe fn launch_size(&self, size: usize) -> Result<(usize, usize), Error> {
        let ctx = self.module.0.device.ctx();
        unsafe {
            let mut unused = 0;
            let mut block_size = 0;
            ctx.cuOccupancyMaxPotentialBlockSize(
                &mut unused,
                &mut block_size,
                self.raw(),
                None,
                0,
                0,
            )
            .check()?;

            let block_size = block_size as usize;
            let grid_size = (size + block_size - 1) / block_size;

            Ok((grid_size, block_size))
        }
    }
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        args: &mut [*mut c_void],
        grid_size: impl Into<KernelSize>,
        block_size: impl Into<KernelSize>,
        shared_size: u32,
    ) -> Result<(), Error> {
        let ctx = self.module.0.device.ctx();
        // let mut unused = 0;
        // let mut block_size = 0;
        // ctx.cuOccupancyMaxPotentialBlockSize(&mut unused, &mut block_size, self.raw(), None, 0, 0)
        //     .check()?;
        // let block_size = block_size as u32;
        //
        // let grid_size = (size as u32 + block_size - 1) / block_size;
        let block_size = block_size.into();
        let grid_size = grid_size.into();

        ctx.cuLaunchKernel(
            self.raw(),
            grid_size.0,
            grid_size.1,
            grid_size.2,
            block_size.0,
            block_size.1,
            block_size.2,
            shared_size,
            **stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
        .check()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct KernelSize(pub u32, pub u32, pub u32);

impl From<u32> for KernelSize {
    fn from(value: u32) -> Self {
        Self(value, 1, 1)
    }
}
impl From<(u32, u32)> for KernelSize {
    fn from(value: (u32, u32)) -> Self {
        Self(value.0, value.1, 1)
    }
}
impl From<(u32, u32, u32)> for KernelSize {
    fn from(value: (u32, u32, u32)) -> Self {
        Self(value.0, value.1, value.2)
    }
}
impl From<usize> for KernelSize {
    fn from(value: usize) -> Self {
        Self(value as _, 1, 1)
    }
}
impl From<(usize, usize)> for KernelSize {
    fn from(value: (usize, usize)) -> Self {
        Self(value.0 as _, value.1 as _, 1)
    }
}
impl From<(usize, usize, usize)> for KernelSize {
    fn from(value: (usize, usize, usize)) -> Self {
        Self(value.0 as _, value.1 as _, value.2 as _)
    }
}
impl From<i32> for KernelSize {
    fn from(value: i32) -> Self {
        Self(value as _, 1, 1)
    }
}
impl From<(i32, i32)> for KernelSize {
    fn from(value: (i32, i32)) -> Self {
        Self(value.0 as _, value.1 as _, 1)
    }
}
impl From<(i32, i32, i32)> for KernelSize {
    fn from(value: (i32, i32, i32)) -> Self {
        Self(value.0 as _, value.1 as _, value.2 as _)
    }
}

#[derive(Debug)]
pub struct Event {
    // stream: Option<Arc<Stream>>,
    device: Arc<InternalDevice>,
    event: CUevent,
}

impl Event {
    pub fn create(device: &Device) -> Result<Self, Error> {
        let ctx = device.ctx();
        unsafe {
            let mut event = std::ptr::null_mut();
            ctx.cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DEFAULT as _)
                .check()?;
            Ok(Self {
                device: device.internal.clone(),
                // stream: None,
                event,
            })
        }
    }
    pub fn raw(&self) -> CUevent {
        self.event
    }
    pub fn synchronize(&self) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuEventSynchronize(self.event).check()?;
        }
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuEventDestroy_v2(self.event).check().unwrap();
        }
    }
}

#[derive(Debug)]
pub struct Buffer {
    device: Arc<InternalDevice>,
    dptr: u64,
    size: usize,
}

impl Buffer {
    pub fn ptr(&self) -> u64 {
        self.dptr
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn uninit(device: &Device, size: usize) -> Result<Self, Error> {
        unsafe {
            let ctx = device.ctx();
            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check()?;
            Ok(Self {
                device: device.internal.clone(),
                dptr,
                size,
            })
        }
    }
    pub fn copy_from_slice(&self, slice: &[u8]) -> Result<(), Error> {
        unsafe {
            assert!(slice.len() <= self.size);
            let ctx = self.device.ctx();

            ctx.cuMemcpyHtoD_v2(self.dptr, slice.as_ptr() as _, slice.len())
                .check()?;
            Ok(())
        }
    }
    pub fn copy_to_host(&self, dst: &mut [u8]) {
        unsafe {
            let ctx = self.device.ctx();
            // assert!(self.size <= dst.len());

            ctx.cuMemcpyDtoH_v2(
                dst.as_mut_ptr() as *mut _,
                self.dptr,
                dst.len().min(self.size),
            )
            .check()
            .unwrap();
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.ctx().cuMemFree_v2(self.dptr).check().unwrap();
        }
    }
}

fn round_pow2(mut x: u32) -> u32 {
    x = x.wrapping_sub(1);
    x |= x.wrapping_shr(1);
    x |= x.wrapping_shr(2);
    x |= x.wrapping_shr(4);
    x |= x.wrapping_shr(8);
    x |= x.wrapping_shr(16);
    x = x.wrapping_add(1);
    x
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    size: usize,
}

impl Info for BufferDesc {
    type Context = Device;

    type Resource = Buffer;

    fn create(info: &Self, ctx: &Self::Context) -> Self::Resource {
        Self::try_create(info, ctx).unwrap()
    }
}
impl TryInfo for BufferDesc {
    type Error = Error;

    fn try_create(info: &Self, ctx: &Self::Context) -> Result<Self::Resource, Self::Error> {
        Buffer::uninit(ctx, info.size)
    }
}

impl Resource for Buffer {
    fn clear(&mut self) {}
}

#[derive(Debug)]
pub struct Texture {
    device: Arc<InternalDevice>,
    array: cuda_rs::CUarray,
    tex: u64,
    shape: [usize; 3],
    dim: usize,
    n_channels: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexutreDesc<'a> {
    pub shape: &'a [usize],
    pub n_channels: u32,
}

impl Texture {
    pub fn array(&self) -> cuda_rs::CUarray {
        self.array
    }
    pub fn tex(&self) -> u64 {
        self.tex
    }
    pub fn n_texels(&self) -> usize {
        self.shape().iter().fold(1, |a, b| a * b)
    }
    pub fn create(device: &Device, desc: &TexutreDesc) -> Result<Self, Error> {
        let shape = desc.shape;
        let n_channels = desc.n_channels;

        let ctx = device.ctx();
        unsafe {
            let mut tex = 0;
            let mut array = std::ptr::null_mut();
            let dim = shape.len();

            log::trace!("Creating texture of dimension {dim}.");

            if dim == 1 || dim == 2 {
                let array_desc = cuda_rs::CUDA_ARRAY_DESCRIPTOR {
                    Width: shape[0],
                    Height: if dim == 1 { 1 } else { shape[1] },
                    Format: cuda_rs::CUarray_format::CU_AD_FORMAT_FLOAT,
                    NumChannels: n_channels as _,
                };
                ctx.cuArrayCreate_v2(&mut array, &array_desc).check()?;
            } else if dim == 3 {
                let array_desc = cuda_rs::CUDA_ARRAY3D_DESCRIPTOR {
                    Width: shape[0],
                    Height: shape[1],
                    Depth: shape[2],
                    Format: cuda_rs::CUarray_format::CU_AD_FORMAT_FLOAT,
                    Flags: 0,
                    NumChannels: n_channels as _,
                };
                ctx.cuArray3DCreate_v2(&mut array, &array_desc).check()?;
            } else {
                unreachable!("Dim cannot be greater than 3");
            };

            let res_desc = cuda_rs::CUDA_RESOURCE_DESC {
                resType: cuda_rs::CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
                res: cuda_rs::CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                    array: cuda_rs::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                        hArray: array,
                    },
                },
                flags: 0,
            };
            let tex_desc = cuda_rs::CUDA_TEXTURE_DESC {
                addressMode: [cuda_rs::CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP; 3],
                filterMode: cuda_rs::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
                flags: 1,
                maxAnisotropy: 1,
                mipmapFilterMode: cuda_rs::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
                ..Default::default()
            };
            let view_desc = cuda_rs::CUDA_RESOURCE_VIEW_DESC {
                format: if n_channels == 1 {
                    cuda_rs::CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_1X32
                } else if n_channels == 2 {
                    cuda_rs::CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_2X32
                } else if n_channels == 4 {
                    cuda_rs::CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_4X32
                } else {
                    panic!("{n_channels} number of channels is not supported!");
                },
                width: shape[0],
                height: if dim >= 2 { shape[1] } else { 1 },
                depth: if dim == 3 { shape[2] } else { 0 },
                ..Default::default()
            };
            ctx.cuTexObjectCreate(&mut tex, &res_desc, &tex_desc, &view_desc)
                .check()
                .unwrap();

            Ok(Self {
                device: device.internal.clone(),
                array,
                tex,
                shape: [
                    *shape.get(0).unwrap_or(&0),
                    *shape.get(1).unwrap_or(&0),
                    *shape.get(2).unwrap_or(&0),
                ],
                dim,
                n_channels,
            })
        }
    }
    pub fn copy_form_buffer(&self, buf: &Buffer, stream: &Stream) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            if self.dim == 1 || self.dim == 2 {
                let pitch = self.shape[0] * self.n_channels as usize * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY2D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcDevice: buf.ptr(),
                    srcPitch: pitch,
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstArray: self.array,
                    WidthInBytes: pitch,
                    Height: if self.dim == 2 { self.shape[1] } else { 1 },
                    ..Default::default()
                };
                ctx.cuMemcpy2DAsync_v2(&op, stream.raw()).check()?;
                Ok(())
            } else {
                let pitch = self.shape[0] * self.n_channels as usize * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY3D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcDevice: buf.ptr(),
                    srcHeight: self.shape[1],
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstArray: self.array,
                    WidthInBytes: pitch,
                    Height: self.shape[1],
                    Depth: self.shape[2],
                    ..Default::default()
                };
                ctx.cuMemcpy3DAsync_v2(&op, stream.raw()).check()?;
                Ok(())
            }
        }
    }
    pub fn copy_to_buffer(&self, buf: &Buffer, stream: &Stream) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            if self.dim == 1 || self.dim == 2 {
                let pitch = self.shape[0] * self.n_channels as usize * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY2D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    srcArray: self.array,
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: buf.ptr(),
                    dstPitch: pitch,
                    WidthInBytes: pitch,
                    Height: if self.dim == 2 { self.shape[1] } else { 1 },
                    ..Default::default()
                };
                ctx.cuMemcpy2DAsync_v2(&op, stream.raw()).check()?;
                Ok(())
            } else {
                let pitch = self.shape[0] * self.n_channels as usize * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY3D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    srcHeight: self.shape[1],
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcArray: self.array,
                    dstDevice: buf.ptr(),
                    WidthInBytes: pitch,
                    Height: self.shape[1],
                    Depth: self.shape[2],
                    ..Default::default()
                };
                ctx.cuMemcpy3DAsync_v2(&op, stream.raw()).check()?;
                Ok(())
            }
        }
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape[0..self.dim]
    }
    pub fn n_channels(&self) -> usize {
        self.n_channels as _
    }
    pub fn ptr(&self) -> u64 {
        self.tex
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuArrayDestroy(self.array).check().unwrap();
            ctx.cuTexObjectDestroy(self.tex).check().unwrap();
        }
    }
}

impl<'a> Info for TexutreDesc<'a> {
    type Context = Device;

    type Resource = Texture;

    fn create(info: &Self, ctx: &Self::Context) -> Self::Resource {
        Self::try_create(info, ctx).unwrap()
    }
}

impl<'a> TryInfo for TexutreDesc<'a> {
    type Error = Error;

    fn try_create(info: &Self, ctx: &Self::Context) -> Result<Self::Resource, Self::Error> {
        Texture::create(ctx, info)
    }
}

impl Resource for Texture {
    fn clear(&mut self) {}
}
