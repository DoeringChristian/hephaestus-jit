use crate::ir::{self, IR};
use crate::trace;
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
