use crate::backend;
use crate::backend::Device;
use crate::data::Data;
use crate::extent::Extent;
use crate::vartype::VarType;
use crate::{compiler, ir, op, trace};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct GraphBuilder {
    buffers: Vec<trace::VarId>,
    id2buffer: HashMap<trace::VarId, BufferId>,
    textures: Vec<trace::VarId>,
    id2texture: HashMap<trace::VarId, TextureId>,
    accels: Vec<trace::VarId>,
    id2accel: HashMap<trace::VarId, AccelId>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn push_buffer(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> BufferId {
        // TODO: use better method to get VarRef
        *self.id2buffer.entry(id).or_insert_with(|| {
            let buffer_id = BufferId(self.buffers.len());
            self.buffers.push(id);
            buffer_id
        })
    }
    pub fn push_texture(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> TextureId {
        // TODO: use better method to get VarRef
        *self.id2texture.entry(id).or_insert_with(|| {
            let texture_id = TextureId(self.textures.len());
            self.textures.push(id);
            texture_id
        })
    }
    pub fn push_accel(&mut self, trace: &mut trace::Trace, id: trace::VarId) -> AccelId {
        *self.id2accel.entry(id).or_insert_with(|| {
            let accel_id = AccelId(self.accels.len());
            self.accels.push(id);
            accel_id
        })
    }
    pub fn push_pass(&mut self, pass: Pass) -> PassId {
        let id = PassId(self.passes.len());
        self.passes.push(pass);
        id
    }
}

#[derive(Debug, Default, Clone)]
pub struct Env {
    buffers: Vec<Option<backend::Buffer>>,
    textures: Vec<Option<backend::Texture>>,
    accels: Vec<Option<backend::Accel>>,
}
impl Env {
    pub fn buffer(&self, id: BufferId) -> &backend::Buffer {
        self.buffers[id.0].as_ref().unwrap()
    }
    pub fn texture(&self, id: TextureId) -> &backend::Texture {
        self.textures[id.0].as_ref().unwrap()
    }
    pub fn accel(&self, id: AccelId) -> &backend::Accel {
        self.accels[id.0].as_ref().unwrap()
    }
}

#[derive(Debug)]
pub struct Graph {
    pub passes: Vec<Pass>,
    pub buffer_descs: Vec<BufferDesc>,
    pub texture_descs: Vec<TextureDesc>,
    pub accel_descs: Vec<AccelDesc>,
    pub env: Env,
    pub schedule: Vec<trace::VarRef>,
}

impl Graph {
    pub fn buffer_ids(&self) -> impl Iterator<Item = BufferId> {
        (0..self.buffer_descs.len()).map(|i| BufferId(i))
    }
    pub fn buffer_desc(&self, buffer_id: BufferId) -> &BufferDesc {
        &self.buffer_descs[buffer_id.0]
    }

    pub fn texture_ids(&self) -> impl Iterator<Item = TextureId> {
        (0..self.buffer_descs.len()).map(|i| TextureId(i))
    }
    pub fn texture_desc(&self, texture_id: TextureId) -> &TextureDesc {
        &self.texture_descs[texture_id.0]
    }

    pub fn accel_ids(&self) -> impl Iterator<Item = AccelId> {
        (0..self.buffer_descs.len()).map(|i| AccelId(i))
    }
    pub fn accel_desc(&self, accel_id: AccelId) -> &AccelDesc {
        &self.accel_descs[accel_id.0]
    }

    pub fn n_passes(&self) -> usize {
        self.passes.len()
    }
    pub fn launch(&mut self, device: &backend::Device) {
        // Capture Environment
        // TODO: it would be nice if we could unify Buffer, Texture and Accel under some common
        // resource typ.
        // This would however also mean changes to the IR representation
        let mut env = Env::default();
        trace::with_trace(|trace| {
            env.buffers = self
                .buffer_descs
                .iter()
                .enumerate()
                .map(|(i, desc)| {
                    Some(
                        if let Some(buffer) =
                            trace.get_var(desc.var_id).and_then(|var| var.data.buffer())
                        {
                            buffer.clone()
                        } else if let Some(buffer) = &self.env.buffers[i] {
                            buffer.clone()
                        } else {
                            device.create_buffer(desc.size * desc.ty.size()).unwrap()
                        },
                    )
                })
                .collect();
            env.textures = self
                .texture_descs
                .iter()
                .enumerate()
                .map(|(i, desc)| {
                    Some(
                        if let Some(texture) = trace
                            .get_var(desc.var_id)
                            .and_then(|var| var.data.texture())
                        {
                            texture.clone()
                        } else if let Some(texture) = &self.env.textures[i] {
                            texture.clone()
                        } else {
                            device.create_texture(desc.shape, desc.channels).unwrap()
                        },
                    )
                })
                .collect();
            env.accels = self
                .accel_descs
                .iter()
                .enumerate()
                .map(|(i, desc)| {
                    Some(
                        if let Some(accel) =
                            trace.get_var(desc.var_id).and_then(|var| var.data.accel())
                        {
                            accel.clone()
                        } else if let Some(accel) = &self.env.accels[i] {
                            accel.clone()
                        } else {
                            device.create_accel(&desc.desc).unwrap()
                        },
                    )
                })
                .collect();
        });

        device.execute_graph(self, &env).unwrap();

        // Update output variables and graph input variables
        trace::with_trace(|trace| {
            for (i, buffer) in env.buffers.into_iter().map(|r| r.unwrap()).enumerate() {
                let desc = &self.buffer_descs[i];
                if let Some(var) = trace.get_var_mut(desc.var_id) {
                    var.data = Data::Buffer(buffer.clone());
                }
                if let Some(input_buffer) = &mut self.env.buffers[i] {
                    *input_buffer = buffer;
                }
            }
            for (i, texture) in env.textures.into_iter().map(|r| r.unwrap()).enumerate() {
                let desc = &self.texture_descs[i];
                if let Some(var) = trace.get_var_mut(desc.var_id) {
                    var.data = Data::Texture(texture.clone());
                }
                if let Some(input_texture) = &mut self.env.textures[i] {
                    *input_texture = texture;
                }
            }
            for (i, accel) in env.accels.into_iter().map(|r| r.unwrap()).enumerate() {
                let desc = &self.accel_descs[i];
                if let Some(var) = trace.get_var_mut(desc.var_id) {
                    var.data = Data::Accel(accel.clone());
                }
                if let Some(input_accel) = &mut self.env.accels[i] {
                    *input_accel = accel;
                }
            }
        })

        // Update output variables in trace
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);

#[derive(Debug, Clone, Copy)]
pub struct BufferId(pub usize);
#[derive(Debug, Clone, Copy)]
pub struct TextureId(pub usize);
#[derive(Debug, Clone, Copy)]
pub struct AccelId(pub usize);

#[derive(Default, Debug)]
pub struct Pass {
    pub buffers: Vec<BufferId>,
    pub textures: Vec<TextureId>,
    pub accels: Vec<AccelId>,
    pub size_buffer: Option<BufferId>,
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
    pub ty: VarType,
    pub var_id: trace::VarId,
}
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub shape: [usize; 3],
    pub channels: usize,
    pub var_id: trace::VarId,
}
#[derive(Debug, Clone)]
pub struct AccelDesc {
    pub desc: backend::AccelDesc,
    pub var_id: trace::VarId,
}

/// Might not be the best but we keep references to `trace::VarRef`s arround to ensure the rc is
/// not 0.
///
/// * `trace`: Trace from which the variables come
/// * `refs`: Variable references
///
/// TODO: should we compile for a device?
pub fn compile(trace: &mut trace::Trace, schedule: trace::Schedule) -> Graph {
    dbg!(&schedule);
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
    for group in groups.iter() {
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
                        size_buffer: None,
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
            let pass = Pass {
                buffers,
                textures,
                accels,
                size_buffer: None,
                op: PassOp::Kernel {
                    ir: compiler.ir,
                    size,
                },
            };
            graph_builder.push_pass(pass);
        }
        // Change op to resulting op
        // This should prevent the ir compiler from colecting stuff twice
        for i in group.clone() {
            let id = vars[i].id();
            let var = trace.var_mut(id);
            if var.data.is_buffer() {
                continue;
            }

            var.op = var.op.resulting_op();
        }
    }

    // Collect descriptors and input resources
    let (buffer_descs, buffers): (Vec<_>, Vec<_>) = graph_builder
        .buffers
        .into_iter()
        .map(|id| {
            let var = trace.var_mut(id);
            let size = var.extent.capacity();

            (
                BufferDesc {
                    size,
                    ty: var.ty.clone(),
                    var_id: id,
                },
                var.data.buffer().cloned(),
            )
        })
        .unzip();
    let (texture_descs, textures): (Vec<_>, Vec<_>) = graph_builder
        .textures
        .into_iter()
        .map(|id| {
            let var = trace.var_mut(id);
            let (shape, channels) = var.extent.shape_and_channles();
            (
                TextureDesc {
                    shape,
                    channels,
                    var_id: id,
                },
                var.data.texture().cloned(),
            )
        })
        .unzip();
    let (accel_descs, accels): (Vec<_>, Vec<_>) = graph_builder
        .accels
        .into_iter()
        .map(|id| {
            let var = trace.var_mut(id);
            let desc = var.extent.accel_desc().clone();
            (AccelDesc { desc, var_id: id }, var.data.accel().cloned())
        })
        .unzip();

    // Cleanup
    for group in groups {
        // Clear Dependecies for schedule variables
        for i in group {
            let id = vars[i].id();
            let var = trace.var_mut(id);

            if var.data.is_buffer() {
                continue;
            }

            // Clear dependencies:
            let deps = std::mem::take(&mut trace.entry_mut(id).deps);

            for dep in deps {
                trace.dec_rc(dep);
            }
        }
    }

    let graph = Graph {
        passes: graph_builder.passes,
        buffer_descs,
        texture_descs,
        accel_descs,
        env: Env {
            buffers,
            textures,
            accels,
        },
        schedule: vars,
    };
    graph
}
