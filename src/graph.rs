use crate::backend;
use crate::extent::Extent;
use crate::resource::{BufferDesc, Resource, ResourceDesc, TextureDesc};
use crate::{compiler, ir, op, trace};
use indexmap::IndexMap;

#[derive(Debug, Default)]
pub struct GraphBuilder {
    resources: IndexMap<trace::VarId, ResourceDesc>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn try_push_resource(
        &mut self,
        trace: &trace::Trace,
        id: trace::VarId,
    ) -> Option<ResourceId> {
        if let Some(id) = self.resources.get_index_of(&id) {
            Some(ResourceId(id))
        } else {
            let var = trace.var(id);
            let desc = match var.op.resulting_op() {
                op::Op::Buffer => ResourceDesc::BufferDesc(BufferDesc {
                    size: var.extent.capacity(),
                    ty: var.ty,
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
    pub fn launch(&self, device: &backend::Device) -> backend::Report {
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
                        if let Some(resource) = &self.env.resources[i] {
                            resource.clone()
                        } else {
                            Resource::create(device, desc)
                        },
                    )
                })
                .collect::<Vec<_>>();
        });

        log::trace!("Launching Graph");
        let report = device.execute_graph(self, &env).unwrap();
        log::trace!("Report:\n {report}");

        // Update resources of variables in the trace and graph.
        // NOTE: We might not want the second one
        trace::with_trace(|trace| {
            for (i, resource) in env.resources.into_iter().map(|r| r.unwrap()).enumerate() {
                let (id, desc) = &self.resource_descs[i];
                if let Some(var) = trace.get_var_mut(*id) {
                    var.data = match &resource {
                        Resource::Buffer(buffer) => Resource::Buffer(buffer.clone()),
                        Resource::Texture(texture) => Resource::Texture(texture.clone()),
                        Resource::Accel(accel) => Resource::Accel(accel.clone()),
                        _ => todo!(),
                    }
                }
            }
        });

        report
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);

#[derive(Debug, Clone, Copy)]
pub struct ResourceId(pub usize);

#[derive(Default, Debug)]
pub struct Pass {
    pub resources: Vec<ResourceId>,
    pub size_buffer: Option<ResourceId>,
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
pub fn compile(trace: &mut trace::Trace, schedule: &trace::Schedule) -> Graph {
    let mut vars = schedule.vars.iter().map(|r| r.id()).collect::<Vec<_>>();
    let groups = schedule.groups.clone();

    // Subdivide groups by extent
    let groups = groups
        .iter()
        .flat_map(|group| {
            vars[group.clone()].sort_by(|id0, id1| {
                trace
                    .var(*id0)
                    .extent
                    .partial_cmp(&trace.var(*id1).extent)
                    .unwrap()
            });

            let mut groups = vec![];
            let mut size = Extent::default();
            let mut start = group.start;

            for i in group.clone() {
                if trace.var(vars[i]).extent != size {
                    let end = i + 1;
                    if start != end {
                        groups.push(start..end);
                    }
                    size = trace.var(vars[i]).extent.clone();
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

    // We can now insert the variables as well as the
    let mut graph_builder = GraphBuilder::default();
    for group in groups.iter() {
        let first_id = vars[group.start];
        let first_var = trace.var(first_id);
        // Note: we know, that all variables in a group have the same size
        // TODO: validate, that the the size_buffer is a buffer
        let size_buffer = match first_var.extent {
            Extent::DynSize { size: size_dep, .. } => {
                graph_builder.try_push_resource(trace, size_dep)
            }
            _ => None,
        };

        let pass = if first_var.op.is_device_op() {
            // Handle Device Ops (precompiled)
            assert_eq!(group.len(), 1);
            let id = vars[group.start];

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

                    Pass {
                        resources,
                        size_buffer,
                        op: PassOp::DeviceOp(op),
                    }
                }
                _ => todo!(),
            }
        } else {
            // Handle Kernel Ops (compile)
            let mut compiler = compiler::Compiler::default();

            compiler.collect_vars(trace, group.clone().map(|i| vars[i]));

            let resources = compiler
                .buffers
                .into_iter()
                .chain(compiler.textures.into_iter())
                .chain(compiler.accels.into_iter())
                .flat_map(|id| graph_builder.try_push_resource(trace, id))
                .collect::<Vec<_>>();
            let size = trace.var(vars[group.start]).extent.capacity();
            Pass {
                resources,
                size_buffer,
                op: PassOp::Kernel {
                    ir: compiler.ir,
                    size,
                },
            }
        };

        graph_builder.push_pass(pass);
        // Change op to resulting op
        // This should prevent the ir compiler from colecting stuff twice
        for i in group.clone() {
            let id = vars[i];
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
            let id = vars[i];
            let var = trace.var_mut(id);

            if var.data.is_buffer() {
                continue;
            }

            // Clear dependencies:
            let deps = std::mem::take(&mut trace.var_mut(id).deps);

            for dep in deps {
                trace.dec_rc(dep);
            }
        }
    }

    let graph = Graph {
        passes: graph_builder.passes,
        resource_descs,
        env: Env { resources },
    };
    graph
}
