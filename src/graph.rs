use crate::backend;
use crate::backend::Device;
use crate::data::Data;
use crate::extent::Extent;
use crate::vartype::VarType;
use crate::{compiler, ir, op, trace};
use indexmap::IndexMap;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Resource {
    Buffer(backend::Buffer),
    Texture(backend::Texture),
    Accel(backend::Accel),
}
#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub size: usize,
    pub ty: VarType,
}
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub shape: [usize; 3],
    pub channels: usize,
}
#[derive(Debug)]
pub enum ResourceDesc {
    BufferDesc(BufferDesc),
    TextureDesc(TextureDesc),
    AccelDesc(backend::AccelDesc),
}

#[derive(Debug, Default)]
pub struct GraphBuilder {
    resources: IndexMap<trace::VarId, ResourceDesc>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn try_push_resource(
        &mut self,
        trace: &mut trace::Trace,
        id: trace::VarId,
    ) -> Option<ResourceId> {
        if let Some(id) = self.resources.get_index_of(&id) {
            Some(ResourceId(id))
        } else {
            let var = trace.var(id);
            let desc = match var.op.resulting_op() {
                op::Op::Buffer => ResourceDesc::BufferDesc(BufferDesc {
                    size: var.extent.capacity(),
                    ty: var.ty.clone(),
                }),
                op::Op::Texture => {
                    let (shape, channels) = var.extent.shape_and_channles();

                    ResourceDesc::TextureDesc(TextureDesc { shape, channels })
                }
                op::Op::Accel => ResourceDesc::AccelDesc(var.extent.accel_desc().clone()),
                _ => return None,
            };
            Some(ResourceId(self.resources.insert_full(id, desc).0))
        }
    }
    pub fn push_pass(&mut self, pass: Pass) -> PassId {
        let id = PassId(self.passes.len());
        self.passes.push(pass);
        id
    }
}

#[derive(Debug, Default, Clone)]
pub struct Env {
    resources: Vec<Option<Resource>>,
}
impl Env {
    pub fn buffer(&self, id: ResourceId) -> Option<&backend::Buffer> {
        match self.resources[id.0].as_ref().unwrap() {
            Resource::Buffer(buffer) => Some(buffer),
            _ => None,
        }
    }
    pub fn texture(&self, id: ResourceId) -> Option<&backend::Texture> {
        match self.resources[id.0].as_ref().unwrap() {
            Resource::Texture(texture) => Some(texture),
            _ => None,
        }
    }
    pub fn accel(&self, id: ResourceId) -> Option<&backend::Accel> {
        match self.resources[id.0].as_ref().unwrap() {
            Resource::Accel(accel) => Some(accel),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    pub passes: Vec<Pass>,
    pub resource_descs: Vec<(trace::VarId, ResourceDesc)>,
    pub env: Env,
    pub schedule: Vec<trace::VarRef>,
}

impl Graph {
    pub fn buffer_desc(&self, id: ResourceId) -> &BufferDesc {
        match &self.resource_descs[id.0].1 {
            ResourceDesc::BufferDesc(desc) => desc,
            _ => todo!(),
        }
    }
    pub fn texture_descs(&self, id: ResourceId) -> &TextureDesc {
        match &self.resource_descs[id.0].1 {
            ResourceDesc::TextureDesc(desc) => desc,
            _ => todo!(),
        }
    }
    pub fn accel_desc(&self, id: ResourceId) -> &backend::AccelDesc {
        match &self.resource_descs[id.0].1 {
            ResourceDesc::AccelDesc(desc) => desc,
            _ => todo!(),
        }
    }

    pub fn n_passes(&self) -> usize {
        self.passes.len()
    }
    pub fn launch(&mut self, device: &backend::Device) {
        // Capture Environment
        let mut env = Env::default();
        trace::with_trace(|trace| {
            env.resources = self
                .resource_descs
                .iter()
                .enumerate()
                .map(|(i, (id, desc))| {
                    Some(
                        // A resource might be captured from different locations
                        // 1. The trace of this thread (if the variable is still alive)
                        // 2. The resources maintained by this graph
                        // 3. Creating new resources
                        if let Some(resource) = trace.get_var(*id).and_then(|var| match desc {
                            ResourceDesc::BufferDesc(_) => {
                                Some(Resource::Buffer(var.data.buffer().cloned()?))
                            }
                            ResourceDesc::TextureDesc(_) => {
                                Some(Resource::Texture(var.data.texture().cloned()?))
                            }
                            ResourceDesc::AccelDesc(_) => {
                                Some(Resource::Accel(var.data.accel().cloned()?))
                            }
                        }) {
                            resource
                        } else if let Some(resource) = &self.env.resources[i] {
                            resource.clone()
                        } else {
                            match desc {
                                ResourceDesc::BufferDesc(desc) => Resource::Buffer(
                                    device.create_buffer(desc.size * desc.ty.size()).unwrap(),
                                ),
                                ResourceDesc::TextureDesc(desc) => Resource::Texture(
                                    device.create_texture(desc.shape, desc.channels).unwrap(),
                                ),
                                ResourceDesc::AccelDesc(desc) => {
                                    Resource::Accel(device.create_accel(desc).unwrap())
                                }
                            }
                        },
                    )
                })
                .collect::<Vec<_>>();
        });

        device.execute_graph(self, &env).unwrap();

        // Update data of output variables and graph input variables.
        trace::with_trace(|trace| {
            for (i, resource) in env.resources.into_iter().map(|r| r.unwrap()).enumerate() {
                let (id, desc) = &self.resource_descs[i];
                if let Some(var) = trace.get_var_mut(*id) {
                    var.data = match &resource {
                        Resource::Buffer(buffer) => Data::Buffer(buffer.clone()),
                        Resource::Texture(texture) => Data::Texture(texture.clone()),
                        Resource::Accel(accel) => Data::Accel(accel.clone()),
                    }
                }
                if let Some(input_resource) = &mut self.env.resources[i] {
                    *input_resource = resource
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
#[derive(Debug, Clone, Copy)]
pub struct ResourceId(pub usize);

#[derive(Default, Debug)]
pub struct Pass {
    pub resources: Vec<ResourceId>,
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

                    // TODO: Improve the readability here. atm. we are pushing all the
                    // dependenceis into multiple vecs starting with [id]
                    let resources = [id]
                        .iter()
                        .chain(deps.iter())
                        .flat_map(|id| graph_builder.try_push_resource(trace, *id))
                        .collect::<Vec<_>>();

                    graph_builder.push_pass(Pass {
                        resources,
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

            let resources = compiler
                .buffers
                .into_iter()
                .chain(compiler.textures.into_iter())
                .chain(compiler.accels.into_iter())
                .flat_map(|id| graph_builder.try_push_resource(trace, id))
                .collect::<Vec<_>>();
            let size = trace.var(vars[group.start].id()).extent.size();
            let pass = Pass {
                resources,
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
    let resource_descs = graph_builder.resources.into_iter().collect::<Vec<_>>();

    let resources = resource_descs
        .iter()
        .map(|(id, desc)| -> Option<Resource> {
            let var = trace.var(*id);
            match desc {
                ResourceDesc::BufferDesc(_) => Some(Resource::Buffer(var.data.buffer().cloned()?)),
                ResourceDesc::TextureDesc(_) => {
                    Some(Resource::Texture(var.data.texture().cloned()?))
                }
                ResourceDesc::AccelDesc(_) => Some(Resource::Accel(var.data.accel().cloned()?)),
            }
        })
        .collect::<Vec<_>>();

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
        resource_descs,
        env: Env { resources },
        schedule: vars,
    };
    graph
}
