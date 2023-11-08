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
    pub env: Env,
    visited: HashMap<trace::VarId, ir::VarId>,
}

impl Compiler {
    pub fn collect_vars(&mut self, trace: &Trace, ids: impl IntoIterator<Item = trace::VarId>) {
        for id in ids {
            let src = self.collect(trace, id);

            let var = trace.var(id);
            if var.ty.size() == 0 {
                continue;
            }

            let buffer_id = self.env.push_buffer(id);
            let dst = self.ir.push_var(
                ir::Var {
                    op: Op::Buffer,
                    ty: var.ty.clone(),
                    data: buffer_id as _,
                    ..Default::default()
                },
                [],
            );
            let idx = self.ir.push_var(
                ir::Var {
                    op: Op::Index,
                    ty: VarType::U32,
                    ..Default::default()
                },
                [],
            );
            self.ir.push_var(
                ir::Var {
                    op: Op::Scatter,
                    ty: VarType::Void,
                    ..Default::default()
                },
                [dst, src, idx],
            );
        }
        self.ir.n_buffers = self.env.buffers.len();
    }
    pub fn collect(&mut self, trace: &trace::Trace, id: trace::VarId) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.var(id);

        let id = match var.op {
            Op::Gather => {
                let src = self.collect_data(trace, var.deps[0]);
                let idx = self.collect(trace, var.deps[1]);
                self.ir.push_var(
                    ir::Var {
                        op: var.op,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    [src, idx],
                )
            }
            Op::Scatter => {
                let dst = self.collect_data(trace, var.deps[0]);
                let src = self.collect(trace, var.deps[1]);
                let idx = self.collect(trace, var.deps[2]);
                self.ir.push_var(
                    ir::Var {
                        op: var.op,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    [dst, src, idx],
                )
            }
            Op::Literal => self.ir.push_var(
                ir::Var {
                    op: Op::Literal,
                    ty: var.ty.clone(),
                    data: var.data.literal().unwrap(),
                    ..Default::default()
                },
                [],
            ),
            Op::Buffer => self.collect_data(trace, id),
            _ => {
                let deps = var
                    .deps
                    .iter()
                    .map(|id| self.collect(trace, *id))
                    .collect::<Vec<_>>();
                self.ir.push_var(
                    ir::Var {
                        op: var.op,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    deps,
                )
            }
        };
        id
    }
    pub fn collect_data(&mut self, trace: &trace::Trace, id: trace::VarId) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.var(id);

        let buffer_id = self.env.push_buffer(id);
        self.ir.push_var(
            ir::Var {
                op: Op::Buffer,
                ty: var.ty.clone(),
                data: buffer_id as _,
                ..Default::default()
            },
            [],
        )
    }
}

pub struct Graph {}

pub struct Pass {
    ir: IR,
    buffers: Vec<trace::VarId>,
}
