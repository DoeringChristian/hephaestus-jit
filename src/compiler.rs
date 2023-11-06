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
    buffers: Vec<Buffer>,
}
impl Env {
    pub fn push_buffer(&mut self, buffer: Buffer) -> usize {
        let i = self.buffers.len();
        self.buffers.push(buffer);
        i
    }
}

#[derive(Debug, Default)]
pub struct Compiler {
    ir: IR,
    env: Env,
    visited: HashMap<trace::VarId, ir::VarId>,
}

impl Compiler {
    pub fn collect_vars(&mut self, trace: &Trace, ids: &[trace::VarId]) {
        for id in ids {
            let src = self.collect(trace, *id);

            let var = trace.var(*id);
            if var.size == 0 {
                continue;
            }

            let buffer_id = self.env.push_buffer(var.data.buffer().cloned().unwrap());
            let dst = self.ir.push_var(
                ir::Var {
                    op: Op::Buffer,
                    ty: var.ty.clone(),
                    data: buffer_id,
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
            Op::Buffer => {
                let buffer_id = self.env.push_buffer(var.data.buffer().cloned().unwrap());
                self.ir.push_var(
                    ir::Var {
                        op: var.op,
                        ty: var.ty.clone(),
                        data: buffer_id,
                        ..Default::default()
                    },
                    [],
                )
            }
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
}

pub struct Kernel {
    ir: IR,
    buffers: Vec<trace::VarId>,
}

pub fn eval(trace: &mut Trace, schedule: impl IntoIterator<Item = trace::VarId>) {
    let mut schedule = schedule.into_iter().collect::<Vec<_>>();
    // For every scheduled variable (destination) we have to create a new buffer (except if it
    // is void)
    for id in schedule.iter() {
        let var = trace.var(*id);
        // Do not reallocate on scheduled variables (don't yet know if this is right)
        // TODO: better test
        if !var.data.is_buffer() && var.ty.size() > 0 {
            let size = trace.var(*id).size;
            let ty_size = trace.var(*id).ty.size();
            let buffer = trace
                .device
                .as_ref()
                .unwrap()
                .create_buffer(size * ty_size)
                .unwrap();

            let var = trace.var_mut(*id);
            var.data = Data::Buffer(buffer);
        }
    }

    schedule.sort_by(|id0, id1| trace.var(*id0).size.cmp(&trace.var(*id1).size));
    let mut schedule_groups = vec![];

    let mut current = 0;
    for i in 1..schedule.len() {
        let size = trace.var(schedule[i - 1]).size;
        if size != trace.var(schedule[i]).size {
            schedule_groups.push(ScheduleGroup {
                size,
                range: current..i,
            });
            current = i;
        }
    }
    schedule_groups.push(ScheduleGroup {
        size: trace.var(schedule[current]).size,
        range: current..schedule.len(),
    });

    let irs = schedule_groups
        .iter()
        .map(|group| {
            let mut scheduler = Compiler::default();
            scheduler.collect_vars(trace, &schedule[group.range.clone()]);
            (scheduler.ir, scheduler.env)
        })
        .collect::<Vec<_>>();

    for ((ir, env), group) in irs.iter().zip(schedule_groups) {
        trace
            .device
            .as_ref()
            .unwrap()
            .execute_ir(ir, group.size, &env.buffers)
            .unwrap();
    }

    // After executing the kernels, the Ir is cleaned up.
    // To do so, we first decrement the refcount and then set the ParamType to Input and op to
    // Data
    for id in schedule {
        let var = trace.var_mut(id);

        // Set op and type for next kernel:
        var.op = Op::Buffer;
        // var.dirty = false;

        // Clear dependencies:
        let deps = std::mem::take(&mut var.deps);

        for dep in deps {
            trace.dec_rc(dep);
        }
    }
}
