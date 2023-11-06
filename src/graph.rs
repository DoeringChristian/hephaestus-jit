use crate::{ir, trace};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct Env {
    buffers: Vec<trace::VarRef>,
}

#[derive(Default, Debug, Clone)]
pub struct Graph {
    passes: Vec<Pass>,
    buffers: Vec<BufferDesc>,
}

impl Graph {
    pub fn push_pass(&mut self, pass: Pass) -> PassId {
        let id = PassId(self.passes.len());
        self.passes.push(pass);
        id
    }
    pub fn pass(&self, id: PassId) -> &Pass {
        &self.passes[id.0]
    }
    pub fn push_buffer(&mut self, desc: BufferDesc) -> BufferId {
        let id = BufferId(self.buffers.len());
        self.buffers.push(desc);
        id
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);
#[derive(Debug, Clone, Copy)]
pub struct BufferId(usize);

#[derive(Default, Debug, Clone)]
pub struct Pass {
    access: Vec<BufferId>,
    op: Op,
}

#[derive(Default, Debug, Clone)]
pub enum Op {
    #[default]
    None,
    CompiledKernel {
        ir: Arc<ir::IR>,
        size: usize,
    },
}
#[derive(Default, Debug, Clone)]
pub struct BufferDesc {
    size: usize,
}
