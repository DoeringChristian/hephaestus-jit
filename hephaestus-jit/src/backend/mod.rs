mod cuda;
mod vulkan;
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use cuda::{CudaBuffer, CudaDevice};
use vulkan::{VulkanBuffer, VulkanDevice, VulkanTexture};

use crate::graph::Env;
use crate::graph::Graph;
use crate::ir::IR;
use crate::trace;
use crate::vartype::AsVarType;
use crate::vartype::VarType;

use self::vulkan::VulkanAccel;

// TODO: Device Caching
pub fn vulkan(id: usize) -> Device {
    Device::vulkan(id).unwrap()
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    VulkanError(#[from] vulkan::Error),
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
    pub fn as_vulkan(&self) -> Option<&VulkanDevice> {
        match self {
            Device::VulkanDevice(device) => Some(device),
            _ => None,
        }
    }
    pub fn create_buffer(&self, size: usize) -> Result<Buffer> {
        let buffer = match self {
            Device::CudaDevice(device) => Buffer::CudaBuffer(device.create_buffer(size)?),
            Device::VulkanDevice(device) => Buffer::VulkanBuffer(device.create_buffer(size)?),
        };
        Ok(buffer)
    }
    pub fn create_buffer_from_slice(&self, slice: &[u8]) -> Result<Buffer> {
        Ok(match self {
            Device::CudaDevice(device) => {
                Buffer::CudaBuffer(device.create_buffer_from_slice(slice)?)
            }
            Device::VulkanDevice(device) => {
                Buffer::VulkanBuffer(device.create_buffer_from_slice(slice)?)
            }
        })
    }
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture> {
        match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => {
                Ok(Texture::VulkanTexture(device.create_texture(desc)?))
            }
        }
    }
    pub fn create_accel(&self, desc: &AccelDesc) -> Result<Accel> {
        match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => Ok(Accel::VulkanAccel(device.create_accel(desc)?)),
        }
    }
    pub fn execute_graph(&self, graph: &Graph, env: &Env) -> Result<Report> {
        let report = match self {
            Device::CudaDevice(_) => todo!(),
            Device::VulkanDevice(device) => device.execute_graph(graph, env),
        };
        if let Ok(report) = &report {
            report.submit_to_profiler();
        };
        report
    }
}

// TODO: we might not need it to be cloneable
#[derive(Debug, Clone)]
pub enum Buffer {
    CudaBuffer(CudaBuffer),
    VulkanBuffer(VulkanBuffer),
}
impl Buffer {
    pub fn to_host<T: AsVarType>(&self, range: std::ops::Range<usize>) -> Result<Vec<T>> {
        match self {
            Self::CudaBuffer(buffer) => Ok(buffer.to_host(range)?),
            Self::VulkanBuffer(buffer) => Ok(buffer.to_host(range)?),
        }
    }
    pub fn device(&self) -> Device {
        match self {
            Buffer::CudaBuffer(buffer) => Device::CudaDevice(buffer.device().clone()),
            Buffer::VulkanBuffer(buffer) => Device::VulkanDevice(buffer.device().clone()),
        }
    }
    pub fn cuda(&self) -> Option<&CudaBuffer> {
        match self {
            Self::CudaBuffer(buffer) => Some(buffer),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Texture {
    VulkanTexture(VulkanTexture),
}

impl Texture {}

#[derive(Debug, Clone)]
pub enum Accel {
    VulkanAccel(VulkanAccel),
}
impl Accel {}

///
/// This trait represents an interface to a device.
/// It has to be implemented for the device.
///
pub trait BackendDevice: Clone + Send + Sync {
    type Buffer: BackendBuffer;
    type Texture: BackendTexture;
    type Accel: BackendAccel;
    fn create_buffer(&self, size: usize) -> Result<Self::Buffer>;
    fn create_buffer_from_slice(&self, slice: &[u8]) -> Result<Self::Buffer>;
    fn create_texture(&self, desc: &TextureDesc) -> Result<Self::Texture>;
    fn create_accel(&self, desc: &AccelDesc) -> Result<Self::Accel>;
    fn execute_graph(&self, graph: &Graph, env: &crate::graph::Env) -> Result<Report> {
        todo!()
    }
}

pub trait BackendBuffer: Clone + Send + Sync {
    type Device: BackendDevice;
    fn to_host<T: AsVarType>(&self, range: std::ops::Range<usize>) -> Result<Vec<T>>;
    // fn size(&self) -> usize;
    fn device(&self) -> &Self::Device;
}

impl AsRef<CudaBuffer> for Buffer {
    fn as_ref(&self) -> &CudaBuffer {
        match self {
            Buffer::CudaBuffer(buffer) => buffer,
            _ => todo!(),
        }
    }
}

pub trait BackendTexture: Debug + Clone + Send + Sync {
    type Device: BackendDevice;
}

pub trait BackendAccel: Clone + Send + Sync {
    type Device: BackendDevice;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeometryDesc {
    Triangles {
        // TODO: add stride
        n_triangles: usize,
        n_vertices: usize,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AccelDesc {
    pub geometries: Arc<[GeometryDesc]>,
    pub instances: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size: usize,
    pub ty: &'static VarType,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureDesc {
    pub shape: [usize; 3],
    pub channels: usize,
    pub format: &'static VarType,
}

pub struct PassReport {
    pub name: String,
    pub start: std::time::Duration, // duration since start of frame
    pub duration: std::time::Duration,
}

pub struct ExecReport {
    pub cpu_start: std::time::SystemTime,
    pub cpu_duration: std::time::Duration,
    pub passes: Vec<PassReport>,
}

#[derive(Default)]
pub struct Report {
    pub exec: Option<ExecReport>,
}
impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Passes:")?;
        if let Some(exec) = &self.exec {
            for pass in exec.passes.iter() {
                writeln!(
                    f,
                    "\t{name: <50} {duration:?} @ {start:?}",
                    name = pass.name,
                    duration = pass.duration,
                    start = pass.start,
                )?;
            }
        }
        Ok(())
    }
}
impl Report {
    pub fn submit_to_profiler(&self) {
        #[cfg(feature = "profile-with-puffin")]
        if let Some(exec) = &self.exec {
            use profiling::puffin;
            let start_ns = exec
                .cpu_start
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64;
            // let start_ns = puffin::now_ns();
            let mut stream = puffin::Stream::default();

            let scope_details = exec
                .passes
                .iter()
                .map(|pass| puffin::ScopeDetails::from_scope_name(pass.name.clone()))
                .collect::<Vec<_>>();

            let ids = puffin::GlobalProfiler::lock().register_user_scopes(&scope_details);

            for (pass, id) in exec.passes.iter().zip(ids.into_iter()) {
                let start = stream.begin_scope(|| start_ns + pass.start.as_nanos() as i64, id, "");
                stream.end_scope(
                    start.0,
                    start_ns + (pass.start.as_nanos() + pass.duration.as_nanos()) as i64,
                );
            }
            // let stream_info = puffin::StreamInfo::parse(stream).unwrap();
            puffin::GlobalProfiler::lock().report_user_scopes(
                puffin::ThreadInfo {
                    start_time_ns: None,
                    name: "gpu".into(),
                },
                &puffin::StreamInfo {
                    stream,
                    num_scopes: 0,
                    depth: 1,
                    range_ns: (
                        start_ns,
                        start_ns
                            + (exec.passes.last().unwrap().start.as_nanos()
                                + exec.passes.last().unwrap().duration.as_nanos())
                                as i64,
                    ),
                }
                .as_stream_into_ref(), //& stream_info.as_stream_into_ref(),
            );
        }
    }
}
