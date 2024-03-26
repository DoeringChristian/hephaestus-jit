pub mod acceleration_structure;
pub mod buffer;
pub mod device;
pub mod graph;
pub mod image;
pub mod physical_device;
pub mod pipeline;
pub mod pool;
pub mod profiler;
pub mod utils;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{0} {1}")]
    Loading(#[from] ash::LoadingError, std::backtrace::Backtrace),
    #[error("{0} {1}")]
    Vk(#[from] ash::vk::Result, std::backtrace::Backtrace),
    #[error("{0} {1}")]
    Allocation(
        #[from] gpu_allocator::AllocationError,
        std::backtrace::Backtrace,
    ),
    #[error("Could not find a Queue Family, supporting the required features!")]
    QueueFamilyNotFound,
    #[error("Could not find physical device {0}!")]
    PhysicalDeviceNotFound(usize),
}

pub type Result<T> = std::result::Result<T, Error>;
