use crate::backend::Buffer;
use crate::data::Data;
use crate::ir::{self, IR};
use crate::op::Op;
use crate::trace::{self, Trace};
use crate::vartype::VarType;
use std::collections::HashMap;
use std::ops::Range;

#[derive(Debug)]
struct ScheduleGroup {
    size: usize,
    range: Range<usize>,
}

#[derive(Debug, Default)]
pub struct Env {
    pub buffers: Vec<trace::VarId>,
}
impl Env {
    pub fn push_buffer(&mut self, id: trace::VarId) -> usize {
        let i = self.buffers.len();
        self.buffers.push(id);
        i
    }
}

#[derive(Debug, Default)]
pub struct Compiler {
    pub ir: IR,
    visited: HashMap<trace::VarId, ir::VarId>,
    id2buffer: HashMap<trace::VarId, usize>,
    pub buffers: Vec<trace::VarId>,
    id2texture: HashMap<trace::VarId, usize>,
    pub textures: Vec<trace::VarId>,
    id2accel: HashMap<trace::VarId, usize>,
    pub accels: Vec<trace::VarId>,
}

impl Compiler {
    pub fn collect_vars(&mut self, trace: &Trace, ids: impl IntoIterator<Item = trace::VarId>) {
        for id in ids {
            let src = self.collect(trace, id);

            let var = trace.var(id);
            if var.ty.size() == 0 {
                continue;
            }

            let buffer_id = self.push_buffer(id);
            let dst = self.ir.push_var(
                ir::Var {
                    op: ir::Op::BufferRef,
                    ty: var.ty.clone(),
                    data: buffer_id as _,
                    ..Default::default()
                },
                [],
            );
            let idx = self.ir.push_var(
                ir::Var {
                    op: ir::Op::Index,
                    ty: VarType::U32,
                    ..Default::default()
                },
                [],
            );
            self.ir.push_var(
                ir::Var {
                    op: ir::Op::Scatter,
                    ty: VarType::Void,
                    ..Default::default()
                },
                [dst, src, idx],
            );
        }
        self.ir.n_buffers = self.buffers.len();
        self.ir.n_textures = self.textures.len();
        self.ir.n_accels = self.accels.len();
    }
    pub fn collect(&mut self, trace: &trace::Trace, id: trace::VarId) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.var(id);

        let id = match var.op {
            Op::Ref { .. } => {
                // When we hit a ref, we just load it as a ref
                self.collect_data(trace, trace.deps(id)[0])
            }
            Op::Buffer => {
                // When we hit a buffer directly we want to access the elements directly
                let data = self.collect_data(trace, id);
                let idx = self.ir.push_var(
                    ir::Var {
                        op: ir::Op::Index,
                        ty: VarType::U32,
                        ..Default::default()
                    },
                    [],
                );
                self.ir.push_var(
                    ir::Var {
                        op: ir::Op::Gather,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    [data, idx],
                )
            }
            Op::KernelOp(kop) => match kop {
                ir::Op::Literal => self.ir.push_var(
                    ir::Var {
                        op: ir::Op::Literal,
                        ty: var.ty.clone(),
                        data: var.data.literal().unwrap(),
                        ..Default::default()
                    },
                    [],
                ),
                _ => {
                    let deps = trace
                        .deps(id)
                        .into_iter()
                        .map(|id| self.collect(trace, *id))
                        .collect::<Vec<_>>();
                    self.ir.push_var(
                        ir::Var {
                            op: kop,
                            ty: var.ty.clone(),
                            ..Default::default()
                        },
                        deps,
                    )
                }
            },
            _ => todo!(), // Op::Buffer => self.collect_data(trace, id),
        };
        id
    }
    pub fn collect_data(&mut self, trace: &trace::Trace, id: trace::VarId) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.var(id);

        match trace.var(id).op.resulting_op() {
            Op::Buffer => {
                let buffer_id = self.push_buffer(id);
                self.ir.push_var(
                    ir::Var {
                        op: ir::Op::BufferRef,
                        ty: var.ty.clone(),
                        data: buffer_id as _,
                        ..Default::default()
                    },
                    [],
                )
            }
            Op::Texture => {
                let texture_id = self.push_texture(id);
                let dim = trace.var(id).extent.texture_dim();
                self.ir.push_var(
                    ir::Var {
                        op: ir::Op::TextureRef { dim },
                        ty: var.ty.clone(),
                        data: texture_id as _,
                        ..Default::default()
                    },
                    [],
                )
            }
            Op::Accel => {
                let accel_id = self.push_accel(id);
                self.ir.push_var(
                    ir::Var {
                        op: ir::Op::AccelRef,
                        ty: var.ty.clone(),
                        data: accel_id as _,
                        ..Default::default()
                    },
                    [],
                )
            }
            _ => todo!(),
        }
    }
    pub fn push_buffer(&mut self, id: trace::VarId) -> usize {
        *self.id2buffer.entry(id).or_insert_with(|| {
            let buffer_id = self.buffers.len();
            self.buffers.push(id);
            buffer_id
        })
    }
    pub fn push_texture(&mut self, id: trace::VarId) -> usize {
        *self.id2texture.entry(id).or_insert_with(|| {
            let texture_id = self.textures.len();
            self.textures.push(id);
            texture_id
        })
    }
    pub fn push_accel(&mut self, id: trace::VarId) -> usize {
        *self.id2accel.entry(id).or_insert_with(|| {
            let accel_id = self.accels.len();
            self.accels.push(id);
            accel_id
        })
    }
}

pub struct Graph {}

pub struct Pass {
    ir: IR,
    buffers: Vec<trace::VarId>,
}
