use crate::backend;
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
    textures: Vec<TextureDesc>,
    id2texture: HashMap<trace::VarId, TextureId>,
    accels: Vec<AccelDesc>,
    id2accel: HashMap<trace::VarId, AccelId>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn push_buffer(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> BufferId {
        assert_eq!(trace.var(id).op.resulting_op(), op::Op::Buffer);
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
    pub fn push_texture(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> TextureId {
        let (shape, channels) = match trace.var(id).op.resulting_op() {
            op::Op::Texture { shape, channels } => (shape, channels),
            _ => todo!(),
        };
        // TODO: use better method to get VarRef
        *self.id2texture.entry(id).or_insert_with(|| {
            let texture_id = TextureId(self.textures.len());
            self.textures.push(TextureDesc {
                shape,
                channels,
                var: trace.ref_borrow(id),
            });
            texture_id
        })
    }
    pub fn push_accel(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> AccelId {
        // TODO: use better method to get VarRef
        *self.id2accel.entry(id).or_insert_with(|| {
            // TODO: reevaluate method here. We are loading the instances from buffer => struct
            // layout has to be correct.

            let accel_desc = trace
                .var(id)
                .accel_desc
                .clone()
                .expect("Expected to find a Accel Descriptor, accociated with this var!");

            let accel_id = AccelId(self.accels.len());
            self.accels.push(AccelDesc {
                desc: accel_desc,
                var: trace.ref_borrow(id),
            });
            accel_id
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
            textures: self.textures,
            accels: self.accels,
            schedule: vec![],
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Graph {
    pub passes: Vec<Pass>,
    pub buffers: Vec<BufferDesc>,
    pub textures: Vec<TextureDesc>,
    pub accels: Vec<AccelDesc>,
    pub schedule: Vec<trace::VarRef>,
}

impl Graph {
    pub fn buffer_desc(&self, buffer_id: BufferId) -> &BufferDesc {
        &self.buffers[buffer_id.0]
    }
    pub fn buffer<'a>(
        &'a self,
        trace: &'a trace::Trace,
        buffer_id: BufferId,
    ) -> &'a backend::Buffer {
        let desc = self.buffer_desc(buffer_id);
        trace.var(desc.var.id()).data.buffer().unwrap()
    }

    pub fn texture_desc(&self, texture_id: TextureId) -> &TextureDesc {
        &self.textures[texture_id.0]
    }
    pub fn texture<'a>(
        &'a self,
        trace: &'a trace::Trace,
        texture_id: TextureId,
    ) -> &'a backend::Texture {
        let desc = self.texture_desc(texture_id);
        trace.var(desc.var.id()).data.texture().unwrap()
    }

    pub fn accel_desc(&self, accel_id: AccelId) -> &AccelDesc {
        &self.accels[accel_id.0]
    }
    pub fn accel<'a>(&'a self, trace: &'a trace::Trace, accel_id: AccelId) -> &'a backend::Accel {
        let desc = self.accel_desc(accel_id);
        trace.var(desc.var.id()).data.accel().unwrap()
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
            if !var.data.is_storage() {
                match var.op.resulting_op() {
                    op::Op::Buffer => {
                        var.data = Data::Buffer(device.create_buffer(size * ty_size).unwrap())
                    }
                    op::Op::Texture { shape, channels } => {
                        var.data = Data::Texture(device.create_texture(shape, channels).unwrap())
                    }
                    _ => todo!(),
                }
            }
        }
        for pass in self.passes.iter() {
            let buffers = pass
                .buffers
                .iter()
                .map(|id| {
                    let desc = self.buffer_desc(*id);
                    trace.var(desc.var.id()).data.buffer().unwrap()
                })
                .collect::<Vec<_>>();
            match &pass.op {
                PassOp::Kernel { ir, size } => {
                    device.execute_ir(ir, *size, buffers.as_slice()).unwrap();
                }
                _ => todo!(),
            }
        }
    }
    pub fn launch(&self, device: &Device) {
        trace::with_trace(|trace| {
            for desc in self.buffers.iter() {
                log::trace!("{:#?}", trace);
                let var = trace.var_mut(desc.var.id());
                // log::trace!("{var:#?}");

                let size = var.size;
                let ty_size = var.ty.size();
                log::trace!("Creating Buffer for {desc:?} {size:?} {ty_size:?}");
                if !var.data.is_storage() {
                    var.data = Data::Buffer(device.create_buffer(size * ty_size).unwrap());
                }
            }
            for desc in self.textures.iter() {
                let var = trace.var_mut(desc.var.id());

                let (shape, channels) = match var.op {
                    op::Op::Texture { shape, channels } => (shape, channels),
                    _ => todo!(),
                };

                if !var.data.is_storage() {
                    var.data = Data::Texture(device.create_texture(shape, channels).unwrap());
                }
            }
            for desc in self.accels.iter() {
                let var = trace.var_mut(desc.var.id());

                if !var.data.is_storage() {
                    var.data = Data::Accel(device.create_accel(&desc.desc).unwrap());
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
#[derive(Debug, Clone, Copy)]
pub struct TextureId(usize);
#[derive(Debug, Clone, Copy)]
pub struct AccelId(usize);

#[derive(Default, Debug, Clone)]
pub struct Pass {
    pub buffers: Vec<BufferId>,
    pub textures: Vec<TextureId>,
    pub accels: Vec<AccelId>,
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
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub shape: [usize; 3],
    pub channels: usize,
    pub var: trace::VarRef,
}
#[derive(Debug, Clone)]
pub struct AccelDesc {
    pub desc: backend::AccelDesc,
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
    // TODO: Instruction Reordering to acheive better kernel splitting

    // Step 2: Put scheduled variables that are groupable into a group:
    let schedule_set = refs.iter().map(|r| r.id()).collect::<HashSet<_>>();

    // TODO: Might be optimizable by using Vec<Range<usize>> instead of Vec<Vec<VarId>>, pointing
    // into `schedule`
    let mut groups = vec![];
    let mut group = vec![];

    // HashMap, tracking the last write (usually scatter) operation to some variable
    // Mitsuba keeps track using a "dirty" flag in the variable
    // We might want to do some other tests (same size and inserting sync calls in kernel), which
    // could reduce the number of kernels generated
    let mut last_write = HashMap::new();

    for id in topo.iter() {
        let var = trace.var(*id);
        if var.op.is_device_op() {
            // Always split on device op (TODO: find more efficient way)
            groups.push(std::mem::take(&mut group));
            groups.push(vec![*id]);
        } else {
            if let crate::op::Op::Ref { mutable: write } = var.op {
                // Split if ref
                groups.push(std::mem::take(&mut group));
                if write {
                    last_write.insert(var.deps[0], *id);
                }
            } else if var.deps.iter().any(|dep_id| {
                // Split when the accessed variable has a write operation pending
                let split_group = last_write.contains_key(dep_id);
                last_write.remove(dep_id);

                split_group
            }) {
                // Split if trying to access variable, that has been written to
                groups.push(std::mem::take(&mut group));
            }

            if schedule_set.contains(id) {
                group.push(*id);
            }
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
        if trace.var(group[0]).op.is_device_op() {
            // Handle Device Ops
            assert_eq!(group.len(), 1);
            let id = group[0];

            match trace.var(id).op {
                op::Op::DeviceOp(op) => match op {
                    op::DeviceOp::Max => todo!(),
                    _ => {
                        let deps = trace.var(id).deps.clone();

                        let mut buffers = vec![];
                        let mut textures = vec![];
                        let mut accels = vec![];

                        // TODO: Improve the readability here. atm. we are pushing all the
                        // dependenceis into multiple vecs starting with [id]
                        for id in [id].iter().chain(deps.iter()) {
                            match trace.var(*id).op.resulting_op() {
                                op::Op::Buffer => {
                                    buffers.push(graph_builder.push_buffer(trace, *id));
                                }
                                op::Op::Texture { .. } => {
                                    textures.push(graph_builder.push_texture(trace, *id));
                                }
                                op::Op::Accel => {
                                    accels.push(graph_builder.push_accel(trace, *id));
                                }
                                _ => todo!(),
                            };
                        }
                        graph_builder.push_pass(Pass {
                            buffers,
                            textures,
                            accels,
                            op: PassOp::DeviceOp(op),
                        })
                    }
                },
                _ => todo!(),
            };
        } else {
            // Handle Kernel Ops (compile)
            let mut compiler = compiler::Compiler::default();

            compiler.collect_vars(trace, group.iter().cloned());

            let buffers = compiler
                .buffers
                .iter()
                .map(|id| graph_builder.push_buffer(trace, *id))
                .collect::<Vec<_>>();
            let textures = compiler
                .textures
                .iter()
                .map(|id| graph_builder.push_texture(trace, *id))
                .collect::<Vec<_>>();
            let accels = compiler
                .accels
                .iter()
                .map(|id| graph_builder.push_accel(trace, *id))
                .collect::<Vec<_>>();
            let pass = Pass {
                buffers,
                textures,
                accels,
                op: PassOp::Kernel {
                    ir: Arc::new(compiler.ir),
                    size: trace.var(group[0]).size,
                },
            };
            graph_builder.push_pass(pass);
        }

        // Clear Dependecies for schedule variables, so that we don't collect to many in the next
        // iteration
        for id in group {
            let var = trace.var_mut(id);

            if var.data.is_buffer() {
                continue;
            }
            // Set op and type for next kernel:
            var.op = var.op.resulting_op();
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
