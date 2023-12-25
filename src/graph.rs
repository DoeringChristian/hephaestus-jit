use crate::backend;
use crate::backend::Device;
use crate::data::Data;
use crate::extent::Extent;
use crate::vartype::VarType;
use crate::{compiler, ir, op, trace};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug)]
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
    pub fn new() -> Self {
        Self {
            buffers: Default::default(),
            id2buffer: Default::default(),
            textures: Default::default(),
            id2texture: Default::default(),
            accels: Default::default(),
            id2accel: Default::default(),
            passes: Default::default(),
        }
    }
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
    // pub fn build(self) -> Graph {
    //     Graph {
    //         device: self.device,
    //         passes: self.passes,
    //         buffers: self.buffers,
    //         textures: self.textures,
    //         accels: self.accels,
    //         schedule: vec![],
    //     }
    // }
}

#[derive(Debug)]
pub struct Graph {
    pub device: backend::Device,
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
    pub fn buffer<'a>(&'a self, buffer_id: BufferId) -> &'a backend::Buffer {
        &self.buffer_desc(buffer_id).buffer
    }

    pub fn texture_desc(&self, texture_id: TextureId) -> &TextureDesc {
        &self.textures[texture_id.0]
    }
    pub fn texture<'a>(&'a self, texture_id: TextureId) -> &'a backend::Texture {
        &self.texture_desc(texture_id).texture
    }

    pub fn accel_desc(&self, accel_id: AccelId) -> &AccelDesc {
        &self.accels[accel_id.0]
    }
    pub fn accel<'a>(&'a self, accel_id: AccelId) -> &'a backend::Accel {
        &self.accel_desc(accel_id).accel
    }

    pub fn n_passes(&self) -> usize {
        self.passes.len()
    }
    pub fn launch(&self) {
        self.device.execute_graph(self).unwrap();
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
    pub buffer: backend::Buffer,
}
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub shape: [usize; 3],
    pub channels: usize,
    pub texture: backend::Texture,
}
#[derive(Debug, Clone)]
pub struct AccelDesc {
    pub desc: backend::AccelDesc,
    pub accel: backend::Accel,
}

/// Might not be the best but we keep references to `trace::VarRef`s arround to ensure the rc is
/// not 0.
///
/// * `trace`: Trace from which the variables come
/// * `refs`: Variable references
///
/// TODO: should we compile for a device?
pub fn compile(
    trace: &mut trace::Trace,
    schedule: trace::Schedule,
    device: &backend::Device,
) -> Graph {
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
    let mut graph_builder = GraphBuilder::new();
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
        for i in group.clone() {
            let id = vars[i].id();
            let var = trace.var_mut(id);
            if var.data.is_buffer() {
                continue;
            }

            var.op = var.op.resulting_op();
        }
    }

    // Create missing resources
    let buffers = graph_builder
        .buffers
        .into_iter()
        .map(|id| {
            log::trace!("Creating buffer for var {id:?}");
            let var = trace.var_mut(id);
            let size = var.extent.capacity();
            let buffer = if let Some(buffer) = var.data.buffer() {
                buffer.clone()
            } else {
                let ty_size = var.ty.size();
                let buffer = device.create_buffer(size * ty_size).unwrap();
                var.data = Data::Buffer(buffer.clone());
                buffer
            };
            BufferDesc {
                size,
                ty: var.ty.clone(),
                buffer,
            }
        })
        .collect::<Vec<_>>();
    let textures = graph_builder
        .textures
        .into_iter()
        .map(|id| {
            let var = trace.var_mut(id);
            let (shape, channels) = var.extent.shape_and_channles();
            let texture = if let Some(texture) = var.data.texture() {
                texture.clone()
            } else {
                let texture = device.create_texture(shape, channels).unwrap();
                var.data = Data::Texture(texture.clone());
                texture
            };
            TextureDesc {
                shape,
                channels,
                texture,
            }
        })
        .collect::<Vec<_>>();
    let accels = graph_builder
        .accels
        .into_iter()
        .map(|id| {
            let var = trace.var_mut(id);
            let desc = var.extent.accel_desc().clone();

            let accel = if let Some(accel) = var.data.accel() {
                accel.clone()
            } else {
                let accel = device.create_accel(&desc).unwrap();
                var.data = Data::Accel(accel.clone());
                accel
            };
            AccelDesc { desc, accel }
        })
        .collect::<Vec<_>>();

    // Cleanup
    for group in groups {
        // Clear Dependecies for schedule variables, so that we don't collect to many in the next
        // iteration
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
        device: device.clone(),
        passes: graph_builder.passes,
        buffers,
        textures,
        accels,
        schedule: vars,
    };
    graph
}
