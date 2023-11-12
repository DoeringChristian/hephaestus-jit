use crate::backend::Device;
use crate::data::Data;
use crate::{compiler, ir, op, trace};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Default, Debug)]
pub struct GraphBuilder {
    buffers: Vec<BufferDesc>,
    id2buffer: HashMap<trace::VarId, BufferId>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn push_buffer(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> BufferId {
        // TODO: use better method to get VarRef
        *self.id2buffer.entry(id).or_insert_with(|| {
            let buffer_id = BufferId(self.buffers.len());
            self.buffers.push(BufferDesc {
                var: trace.ref_borrow(id),
                size: trace.var(id).size,
            });
            buffer_id
        })
    }
    pub fn push_pass(&mut self, pass: Pass) -> PassId {
        let id = PassId(self.passes.len());
        self.passes.push(pass);
        id
    }
    pub fn build(self) -> Graph {
        Graph {
            passes: self.passes,
            buffers: self.buffers,
            ..Default::default()
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Graph {
    pub passes: Vec<Pass>,
    pub buffers: Vec<BufferDesc>,
    pub schedule: Vec<trace::VarRef>,
}

impl Graph {
    pub fn buffer_desc(&self, buffer_id: BufferId) -> &BufferDesc {
        &self.buffers[buffer_id.0]
    }
    pub fn n_passes(&self) -> usize {
        self.passes.len()
    }
    pub fn launch_slow(&self, device: &Device) {
        trace::with_trace(|t| {
            self._launch_slow(t, device);
        })
    }
    fn _launch_slow(&self, trace: &mut trace::Trace, device: &Device) {
        for desc in self.buffers.iter() {
            let var = trace.var_mut(desc.var.id());

            let size = var.size;
            let ty_size = var.ty.size();
            if var.data.is_none() | var.data.is_literal() {
                var.data = Data::Buffer(device.create_buffer(size * ty_size).unwrap());
            }
        }
        for pass in self.passes.iter() {
            let buffers = pass
                .buffers
                .iter()
                .map(|id| {
                    let desc = self.buffer_desc(*id);
                    trace.var(desc.var.id()).data.buffer().unwrap().clone()
                })
                .collect::<Vec<_>>();
            match &pass.op {
                PassOp::Kernel { ir, size } => {
                    device.execute_ir(ir, *size, &buffers).unwrap();
                }
                _ => todo!(),
            }
        }
    }
    pub fn launch(&self, device: &Device) {
        trace::with_trace(|trace| {
            for desc in self.buffers.iter() {
                let var = trace.var_mut(desc.var.id());

                let size = var.size;
                let ty_size = var.ty.size();
                if var.data.is_none() | var.data.is_literal() {
                    var.data = Data::Buffer(device.create_buffer(size * ty_size).unwrap());
                }
            }
            device.execute_graph(trace, self).unwrap();
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);
#[derive(Debug, Clone, Copy)]
pub struct BufferId(usize);

#[derive(Default, Debug, Clone)]
pub struct Pass {
    pub buffers: Vec<BufferId>,
    pub op: PassOp,
}

#[derive(Default, Debug, Clone)]
pub enum PassOp {
    #[default]
    None,
    Kernel {
        ir: Arc<ir::IR>,
        size: usize,
    },
    DeviceOp(op::DeviceOp),
}
#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub size: usize,
    pub var: trace::VarRef,
}

/// Might not be the best but we keep references to `trace::VarRef`s arround to ensure the rc is
/// not 0.
///
/// * `trace`: Trace from which the variables come
/// * `refs`: Variable references
pub fn compile(trace: &mut trace::Trace, refs: Vec<trace::VarRef>) -> Graph {
    // Step 1: Create topological ordering by DFS, making sure to keep order from schedule
    let mut topo = vec![];
    let mut visited = HashMap::<trace::VarId, usize>::default();

    fn visit(
        trace: &trace::Trace,
        visited: &mut HashMap<trace::VarId, usize>,
        topo: &mut Vec<trace::VarId>,
        id: trace::VarId,
    ) {
        if visited.contains_key(&id) {
            return;
        }

        for id in trace.var(id).deps.iter() {
            visit(trace, visited, topo, *id);
        }
        visited.insert(id, topo.len());
        topo.push(id);
    }
    for id in refs.iter().map(|r| r.id()) {
        visit(trace, &mut visited, &mut topo, id);
    }

    // Step 2: Put scheduled variables that are groupable into a group:
    let schedule_set = refs.iter().map(|r| r.id()).collect::<HashSet<_>>();

    let mut groups = vec![];
    let mut group = vec![];

    // This is a bit of a cheat, but Mitsuba does somethig similar
    let mut dirty = HashSet::new();

    for id in topo.iter() {
        let var = trace.var(*id);
        if let crate::op::Op::Ref { mutable: write } = var.op {
            // Split if ref
            groups.push(std::mem::take(&mut group));
            if write {
                dirty.insert(var.deps[0]);
            }
        } else if var.deps.iter().any(|id| {
            let is_dirty = dirty.contains(id);
            dirty.remove(id);
            is_dirty
        }) {
            // Split if trying to access dirty
            groups.push(std::mem::take(&mut group));
        }

        if schedule_set.contains(id) {
            group.push(*id);
        } else {
        }
    }
    if !group.is_empty() {
        groups.push(group);
    }

    // Step 3: subdivide groups by size:

    let groups = groups
        .iter_mut()
        .flat_map(|group| {
            group.sort_by(|id0, id1| trace.var(*id0).size.cmp(&trace.var(*id1).size));
            let groups_iter = group
                .iter()
                .cloned()
                .group_by(|id| trace.var(*id).size.clone());
            groups_iter
                .into_iter()
                .map(|(_, group)| group.collect::<Vec<_>>())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // We can now insert the variables as well as the
    let mut graph_builder = GraphBuilder::default();
    for group in groups {
        let mut compiler = compiler::Compiler::default();

        for id in group.iter() {
            let var = trace.var(*id);
            match var.op {
                op::Op::DeviceOp(dop) => todo!(),
                op::Op::KernelOp(_) => compiler.collect_vars(trace, Some(*id)),
                _ => todo!(),
            }
        }

        let buffers = compiler
            .buffers
            .iter()
            .map(|id| graph_builder.push_buffer(trace, *id))
            .collect::<Vec<_>>();
        let pass = Pass {
            buffers,
            op: PassOp::Kernel {
                ir: Arc::new(compiler.ir),
                size: trace.var(group[0]).size,
            },
        };
        graph_builder.push_pass(pass);

        // Clear Dependecies for schedule variables, so that we don't collect to many in the next
        // iteration
        for id in group {
            use crate::op;

            let var = trace.var_mut(id);

            if var.data.is_buffer() {
                continue;
            }
            // Set op and type for next kernel:
            var.op = op::Op::Buffer;
            // var.dirty = false;

            // Clear dependencies:
            let deps = std::mem::take(&mut var.deps);

            for dep in deps {
                trace.dec_rc(dep);
            }
        }
    }

    let mut graph = graph_builder.build();
    graph.schedule = refs;
    graph
}
