use crate::backend::Device;
use crate::data::Data;
use crate::{compiler, ir, trace};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Default, Debug)]
pub struct GraphBuilder {
    buffers: Vec<BufferDesc>,
    id2buffer: HashMap<trace::VarId, BufferId>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn push_buffer(&mut self, trace: &trace::Trace, id: trace::VarId) -> BufferId {
        *self.id2buffer.entry(id).or_insert_with(|| {
            let buffer_id = BufferId(self.buffers.len());
            self.buffers.push(BufferDesc {
                id,
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
    passes: Vec<Pass>,
    buffers: Vec<BufferDesc>,
    schedule: Vec<trace::VarRef>,
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
            let var = trace.var_mut(desc.id);

            let size = var.size;
            let ty_size = var.ty.size();
            if var.data.is_none() {
                var.data = Data::Buffer(device.create_buffer(size * ty_size).unwrap());
            }
        }
        for pass in self.passes.iter() {
            let buffers = pass
                .buffers
                .iter()
                .map(|id| {
                    let desc = self.buffer_desc(*id);
                    trace.var(desc.id).data.buffer().unwrap().clone()
                })
                .collect::<Vec<_>>();
            match &pass.op {
                Op::CompiledKernel { ir, size } => {
                    device.execute_ir(ir, *size, &buffers).unwrap();
                }
                _ => todo!(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);
#[derive(Debug, Clone, Copy)]
pub struct BufferId(usize);

#[derive(Default, Debug, Clone)]
pub struct Pass {
    buffers: Vec<BufferId>,
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
#[derive(Debug, Clone)]
pub struct BufferDesc {
    size: usize,
    id: trace::VarId,
}

/// Might not be the best but we keep references to `trace::VarRef`s arround to ensure the rc is
/// not 0.
///
/// * `trace`: Trace from which the variables come
/// * `refs`: Variable references
pub fn compile(trace: &mut trace::Trace, refs: Vec<trace::VarRef>) -> Graph {
    let mut schedule = refs.iter().map(|r| r.id()).collect::<Vec<_>>();
    /// Test if `larger` depends on `smaller` and the connection between them is broken i.e. the
    /// there is a gather operation between them.
    ///
    /// * `trace`: Trace
    /// * `larger`: The variable appearing later in the ssa
    /// * `smaller`: The variable appearing earlier in the ssa
    /// TODO: improve performance by caching
    fn broken_dep(trace: &trace::Trace, larger: trace::VarId, smaller: trace::VarId) -> bool {
        use crate::op::Op;
        let lvar = trace.var(larger);
        let broken = match lvar.op {
            Op::Gather => {
                if lvar.deps[0] == smaller {
                    true
                } else {
                    lvar.deps[1..]
                        .iter()
                        .map(|d| broken_dep(trace, *d, smaller))
                        .fold(false, |a, b| a || b)
                }
            }
            Op::Scatter => {
                if lvar.deps[0] == smaller {
                    true
                } else {
                    lvar.deps[1..]
                        .iter()
                        .map(|d| broken_dep(trace, *d, smaller))
                        .fold(false, |a, b| a || b)
                }
            }
            _ => lvar
                .deps
                .iter()
                .map(|d| broken_dep(trace, *d, smaller))
                .fold(false, |a, b| a || b),
        };
        broken
    }

    // Decides weather two variables can be merged into the same Kernel.
    let mergable = |larger, smaller| {
        let mergable = trace.var(smaller).size == trace.var(larger).size;
        let mergable = mergable & !broken_dep(trace, larger, smaller);
        mergable
    };
    let mut groups = vec![];
    while !schedule.is_empty() {
        let mut group = vec![schedule.pop().unwrap()];

        // Add all mergable elements to group, taking them from the schedule
        schedule.retain(|smaller| {
            let mergable = group
                .iter()
                .map(|larger| mergable(*larger, *smaller))
                .fold(true, |a, b| a && b);
            if mergable {
                group.push(*smaller);
            }
            !mergable
        });
        // We should now have a group of variables that can be compiled into the same kernel.
        groups.push(group);
    }

    // We should now have variables grouped according to weather they can be launched in the same
    // kernel
    // They are sorted in reverse order, and have to be reversed.
    groups.reverse();

    // We can now insert the variables as well as the
    let mut graph_builder = GraphBuilder::default();
    for group in groups {
        let mut compiler = compiler::Compiler::default();
        compiler.collect_vars(trace, group.iter().cloned());

        let buffers = compiler
            .env
            .buffers
            .iter()
            .map(|id| graph_builder.push_buffer(trace, *id))
            .collect::<Vec<_>>();
        let pass = Pass {
            buffers,
            op: Op::CompiledKernel {
                ir: Arc::new(compiler.ir),
                size: trace.var(group[0]).size,
            },
        };
        graph_builder.push_pass(pass);
    }

    let mut graph = graph_builder.build();
    graph.schedule = refs;
    graph
}
