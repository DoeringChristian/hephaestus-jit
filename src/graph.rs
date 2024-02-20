use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::extent::Extent;
use crate::resource::{BufferDesc, Resource, ResourceDesc, TextureDesc};
use crate::vartype::AsVarType;
use crate::{backend, vartype};
use crate::{compiler, ir, op, trace};
use indexmap::IndexMap;

///
/// Specifies a Graph Resource.
/// These can either be parameters, given to a closure or
/// resources captured by the closure or
/// Resources used internally by the graph.
/// In the case where variables, referenced by
/// internal resources still exist at time of execution,
/// they get overriden by the internal resource.
///
pub enum GraphResource {
    Input { idx: usize },
    Output { idx: usize },
    Captured { resource: Resource },
    Internal { id: trace::VarId },
}
// Have to implement debug for snapshot tests to work
impl Debug for GraphResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input { idx } => f.debug_struct("Param").field("idx", idx).finish(),
            Self::Captured { resource } => f
                .debug_struct("Captured")
                .field("resource", resource)
                .finish(),
            Self::Internal { .. } => f.debug_struct("Internal").finish(),
            Self::Output { idx } => f.debug_struct("Output").field("idx", idx).finish(),
        }
    }
}

///
/// Helper struct, used to build a Graph.
/// It manages passes and resource, using an index map.
///
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
    resources: Vec<Resource>,
}
impl Env {
    pub fn buffer(&self, id: ResourceId) -> Option<&backend::Buffer> {
        match &self.resources[id.0] {
            Resource::Buffer(buffer) => Some(buffer),
            _ => None,
        }
    }
    pub fn texture(&self, id: ResourceId) -> Option<&backend::Texture> {
        match &self.resources[id.0] {
            Resource::Texture(texture) => Some(texture),
            _ => None,
        }
    }
    pub fn accel(&self, id: ResourceId) -> Option<&backend::Accel> {
        match &self.resources[id.0] {
            Resource::Accel(accel) => Some(accel),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    pub passes: Vec<Pass>,
    pub resource_descs: Vec<ResourceDesc>,
    pub resources: Vec<GraphResource>,
    pub n_outputs: usize,
}

impl Graph {
    pub fn buffer_desc(&self, id: ResourceId) -> &BufferDesc {
        match &self.resource_descs[id.0] {
            ResourceDesc::BufferDesc(desc) => desc,
            _ => todo!(),
        }
    }
    pub fn texture_descs(&self, id: ResourceId) -> &TextureDesc {
        match &self.resource_descs[id.0] {
            ResourceDesc::TextureDesc(desc) => desc,
            _ => todo!(),
        }
    }
    pub fn accel_desc(&self, id: ResourceId) -> &backend::AccelDesc {
        match &self.resource_descs[id.0] {
            ResourceDesc::AccelDesc(desc) => desc,
            _ => todo!(),
        }
    }

    pub fn n_passes(&self) -> usize {
        self.passes.len()
    }
    pub fn launch(&self, device: &backend::Device) -> backend::Report {
        self.launch_with(device, &[]).0
    }
    pub fn launch_with(
        &self,
        device: &backend::Device,
        inputs: &[&trace::VarRef],
    ) -> (backend::Report, Vec<trace::VarRef>) {
        // Capture Environment
        let mut env = Env::default();
        trace::with_trace(|trace| {
            env.resources = self
                .resources
                .iter()
                .zip(self.resource_descs.iter())
                .map(|(res, desc)| match res {
                    GraphResource::Input { idx } => {
                        let var = trace.var(inputs[*idx].id());
                        var.data.match_and_get(desc).unwrap()
                    }
                    GraphResource::Output { .. } => Resource::create(device, desc),
                    GraphResource::Captured { resource } => resource.clone(),
                    GraphResource::Internal { .. } => Resource::create(device, desc),
                })
                .collect::<Vec<_>>();
        });

        log::trace!("Launching Graph");
        let report = device.execute_graph(self, &env).unwrap();
        log::trace!("Report:\n {report}");

        let mut output = vec![None; self.n_outputs];

        // Update resources of variables in the trace.
        trace::with_trace(|trace| {
            env.resources
                .into_iter()
                .zip(self.resources.iter())
                .zip(self.resource_descs.iter())
                .for_each(|((res, gres), desc)| match gres {
                    GraphResource::Internal { id } => {
                        if let Some(var) = trace.get_var_mut(*id) {
                            var.data = res;
                        }
                    }
                    GraphResource::Output { idx } => {
                        let op = match res {
                            Resource::Buffer(_) => op::Op::Buffer,
                            Resource::Texture(_) => op::Op::Texture,
                            Resource::Accel(_) => op::Op::Accel,
                            _ => todo!(),
                        };
                        let (ty, extent) = match desc {
                            ResourceDesc::BufferDesc(desc) => (desc.ty, Extent::Size(desc.size)),
                            ResourceDesc::TextureDesc(desc) => (
                                f32::var_ty(),
                                Extent::Texture {
                                    shape: desc.shape,
                                    channels: desc.channels,
                                },
                            ),
                            ResourceDesc::AccelDesc(desc) => {
                                (vartype::void(), Extent::Accel(desc.clone()))
                            }
                        };
                        output[*idx] = Some(trace.push_var(trace::Var {
                            op,
                            ty,
                            extent,
                            data: res,
                            ..Default::default()
                        }))
                    }
                    _ => {}
                });
        });

        let output = output.into_iter().collect::<Option<Vec<_>>>().unwrap();

        (report, output)
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
pub fn compile(
    trace: &mut trace::Trace,
    ts: &trace::ThreadState,
    input: &[&trace::VarRef],
    output: &[&trace::VarRef],
) -> Graph {
    let input = input
        .into_iter()
        .enumerate()
        .map(|(i, r)| (r.id(), i))
        .collect::<HashMap<_, _>>();
    let output = output
        .into_iter()
        .enumerate()
        .map(|(i, r)| (r.id(), i))
        .collect::<HashMap<_, _>>();
    let n_outputs = output.len();

    // Get scheduled variables from thread state in order
    let mut vars = ts.scheduled.values().map(|r| r.id()).collect::<Vec<_>>();
    let groups = ts.groups.clone();

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
            thread_local! {
                static COMPILER: RefCell<compiler::Compiler> = Default::default();
            };
            let (resources, ir) = COMPILER.with(|compiler| {
                let mut compiler = compiler.borrow_mut();
                compiler.clear();

                compiler.compile(trace, &vars[group.clone()]);

                let resources = compiler
                    .buffers
                    .iter()
                    .chain(compiler.textures.iter())
                    .chain(compiler.accels.iter())
                    .flat_map(|&id| graph_builder.try_push_resource(trace, id))
                    .collect::<Vec<_>>();

                let ir = std::mem::take(&mut compiler.ir);
                (resources, ir)
            });
            let size = trace.var(vars[group.start]).extent.capacity();
            Pass {
                resources,
                size_buffer,
                op: PassOp::Kernel { ir, size },
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

    let resources = graph_builder
        .resources
        .iter()
        .map(|(id, desc)| {
            if input.contains_key(&id) {
                GraphResource::Input { idx: input[&id] }
            } else if output.contains_key(&id) {
                GraphResource::Output { idx: output[&id] }
            } else {
                let var = trace.var(*id);
                var.data
                    .match_and_get(desc)
                    .map(|resource| GraphResource::Captured { resource })
                    .unwrap_or_else(|| GraphResource::Internal { id: *id })
            }
        })
        .collect::<Vec<_>>();

    let resource_descs = graph_builder
        .resources
        .into_iter()
        .map(|(_, desc)| desc)
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
        resources,
        n_outputs,
    };
    graph
}
