use ash::vk;
use gpu_allocator::MemoryLocation;

use super::buffer::{Buffer, BufferInfo};
use super::device::Device;

pub struct Profiler {
    device: Device,
    query_pool: vk::QueryPool,
    buffer: Buffer,
    next_scope: u32,
    max_scopes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScopeId(u32);

pub struct TimedScope {
    pub start: std::time::Duration,
    pub duration: std::time::Duration,
}

type DurationRange = [u64; 2];

impl Profiler {
    #[profiling::function]
    pub fn new(device: &Device, max_scopes: usize) -> Self {
        let buffer = Buffer::create(
            device,
            BufferInfo {
                size: max_scopes * std::mem::size_of::<DurationRange>(),
                alignment: 0,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuToCpu,
            },
        );

        let pool_info = vk::QueryPoolCreateInfo {
            query_type: vk::QueryType::TIMESTAMP,
            query_count: max_scopes as u32 * 2,
            ..Default::default()
        };

        let query_pool = unsafe { device.create_query_pool(&pool_info, None).unwrap() };

        Self {
            device: device.clone(),
            buffer,
            query_pool,
            next_scope: 0,
            max_scopes,
        }
    }
    pub fn begin_frame(&mut self, cb: vk::CommandBuffer) {
        unsafe {
            self.device
                .cmd_reset_query_pool(cb, self.query_pool, 0, self.max_scopes as u32 * 2)
        };
    }
    pub fn begin_scope(&mut self, cb: vk::CommandBuffer) -> ScopeId {
        let scope_idx = self.next_scope;
        self.next_scope += 1;

        unsafe {
            self.device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                scope_idx * 2,
            )
        };
        ScopeId(scope_idx)
    }
    pub fn end_scope(&mut self, cb: vk::CommandBuffer, scope_id: ScopeId) {
        unsafe {
            self.device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                scope_id.0 * 2 + 1,
            )
        }
    }
    pub fn end_frame(&mut self, cb: vk::CommandBuffer) {
        let query_count = self.next_scope * 2;
        unsafe {
            self.device.cmd_copy_query_pool_results(
                cb,
                self.query_pool,
                0,
                query_count,
                self.buffer.vk(),
                0,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
    }
    pub fn report(self) -> Vec<TimedScope> {
        // Copy query to buffer

        let ns_per_tick = self
            .device
            .physical_device
            .properties
            .limits
            .timestamp_period as f64;

        let report: Vec<DurationRange> =
            bytemuck::cast_slice(self.buffer.mapped_slice())[..self.next_scope as usize].to_vec();

        let start_frame = report[0][0];

        let report = report
            .into_iter()
            .map(|[start_scope, end_scope]| {
                let start = std::time::Duration::from_nanos(
                    ((start_scope - start_frame) as f64 * ns_per_tick) as u64,
                );
                let duration = std::time::Duration::from_nanos(
                    ((end_scope - start_scope) as f64 * ns_per_tick) as u64,
                );
                TimedScope { start, duration }
            })
            .collect::<Vec<_>>();

        report
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
        }
    }
}
