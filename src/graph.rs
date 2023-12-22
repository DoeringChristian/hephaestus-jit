use crate::backend;
use crate::backend::Device;
use crate::data::Data;
use crate::extent::Extent;
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
                size: trace.var(id).extent.capacity(),
            });
            buffer_id
        })
    }
    pub fn push_texture(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> TextureId {
        let (shape, channels) = trace.var(id).extent.shape_and_channles();
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

            let accel_desc = trace.var(id).extent.accel_desc().clone();

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

#[derive(Default, Debug)]
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
    pub fn launch(&self, device: &Device) {
        trace::with_trace(|trace| {
            for desc in self.buffers.iter() {
                log::trace!("{:#?}", trace);
                let var = trace.var_mut(desc.var.id());
                // log::trace!("{var:#?}");

                let size = var.extent.capacity();
                let ty_size = var.ty.size();
                log::trace!("Creating Buffer for {desc:?} {size:?} {ty_size:?}");
                if !var.data.is_storage() {
                    var.data = Data::Buffer(device.create_buffer(size * ty_size).unwrap());
                }
            }
            for desc in self.textures.iter() {
                let var = trace.var_mut(desc.var.id());

                let (shape, channels) = var.extent.shape_and_channles();

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

#[derive(Default, Debug)]
pub struct Pass {
    pub buffers: Vec<BufferId>,
    pub textures: Vec<TextureId>,
    pub accels: Vec<AccelId>,
    // pub size: Option<BufferId>,
    pub op: PassOp,
}

#[derive(Default, Debug)]
pub enum PassOp {
    #[default]
    None,
    Kernel {
        ir: ir::IR,
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
pub fn compile(trace: &mut trace::Trace, schedule: trace::Schedule) -> Graph {
    let trace::Schedule {
        mut vars,
        mut groups,
        ..
    } = schedule;

    // Split groups into groups by extent
    let groups = groups
        .iter_mut()
        .flat_map(|group| {
            vars[group.clone()].sort_by(|id0, id1| {
                trace
                    .var(id0.id())
                    .extent
                    .partial_cmp(&trace.var(id1.id()).extent)
                    .unwrap()
            });

            let mut groups = vec![];
            let mut size = Extent::default();
            let mut start = group.start;

            for i in group.clone() {
                if trace.var(vars[i].id()).extent != size {
                    let end = i + 1;
                    if start != end {
                        groups.push(start..end);
                    }
                    size = trace.var(vars[i].id()).extent.clone();
                    start = end;
                }
            }
            let end = group.end;
            if start != end {
                groups.push(start..end);
            }
            groups
        })
        .collect::<Vec<_>>();
    // dbg!(&groups);
    // dbg!(&vars);

    // We can now insert the variables as well as the
    let mut graph_builder = GraphBuilder::default();
    for group in groups {
        if trace.var(vars[group.start].id()).op.is_device_op() {
            // Handle Device Ops (precompiled)
            assert_eq!(group.len(), 1);
            let id = vars[group.start].id();

            let var = trace.var(id);
            match var.op {
                op::Op::DeviceOp(op) => {
                    let deps = trace.deps(id).to_vec();

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
                            _ => {
                                log::trace!("No valid resulting operation, skipping!");
                            }
                        };
                    }
                    graph_builder.push_pass(Pass {
                        buffers,
                        textures,
                        accels,
                        // size: None,
                        op: PassOp::DeviceOp(op),
                    })
                }
                _ => todo!(),
            };
        } else {
            // Handle Kernel Ops (compile)
            let mut compiler = compiler::Compiler::default();

            compiler.collect_vars(trace, group.clone().map(|i| vars[i].id()));

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
            let size = trace.var(vars[group.start].id()).extent.size();
            dbg!(size);
            let pass = Pass {
                buffers,
                textures,
                accels,
                // size: None,
                op: PassOp::Kernel {
                    ir: compiler.ir,
                    size,
                },
            };
            graph_builder.push_pass(pass);
        }

        // Clear Dependecies for schedule variables, so that we don't collect to many in the next
        // iteration
        for i in group {
            let id = vars[i].id();
            let var = trace.var_mut(id);

            if var.data.is_buffer() {
                continue;
            }
            // Set op and type for next kernel:
            var.op = var.op.resulting_op();
            // var.dirty = false;

            // Clear dependencies:
            let deps = std::mem::take(&mut trace.entry_mut(id).deps);

            for dep in deps {
                trace.dec_rc(dep);
            }
        }
    }

    let mut graph = graph_builder.build();
    graph.schedule = vars;
    graph
}
