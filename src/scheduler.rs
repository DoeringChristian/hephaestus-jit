use crate::data::Data;
use crate::ir::{self, IR};
use crate::trace::{self, Trace};
use std::collections::HashMap;
use std::ops::Range;

#[derive(Debug)]
struct ScheduleGroup {
    size: usize,
    range: Range<usize>,
}

#[derive(Debug, Default)]
pub struct Scheduler {
    ir: IR,
    visited: HashMap<trace::VarId, ir::VarId>,
}

impl Scheduler {
    pub fn collect(&mut self, trace: &trace::Trace, id: trace::VarId) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.var(id);

        let id = match var.op {
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

pub fn eval(trace: &mut Trace, refs: &[&trace::VarRef]) {
    let mut schedule = refs.iter().map(|r| r.0).collect::<Vec<_>>();
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
    dbg!(&schedule_groups);
}
