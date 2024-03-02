use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;

use crate::extent::Extent;
use crate::resource::{BufferDesc, Resource, ResourceDesc, TextureDesc};
use crate::vartype::AsVarType;
use crate::{backend, vartype};
use crate::{compiler, ir, op, trace};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;

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
    Input {
        idx: usize,
        out_idx: Option<usize>, // NOTE: might also be returned
    },
    Output {
        idx: usize,
    },
    // Using strong reference for captured variables
    Captured {
        r: trace::VarRef,
        refs: Vec<trace::VarRef>,
        out_idx: Option<usize>, // NOTE: might also be returned
    },
    // Using weak reference for internal variables
    Internal {
        id: trace::VarId,
        ids: Vec<trace::VarId>,
    },
}
// Have to implement debug for snapshot tests to work
impl Debug for GraphResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input { idx, out_idx } => f
                .debug_struct("Input")
                .field("idx", idx)
                .field("out_idx", out_idx)
                .finish(),
            Self::Output { idx } => f.debug_struct("Output").field("idx", idx).finish(),
            Self::Captured { .. } => f.debug_struct("Captured").finish(),
            Self::Internal { .. } => f.debug_struct("Internal").finish(),
        }
    }
}

///
/// Helper struct, used to build a Graph.
/// It manages passes and resource, using an index map.
///
#[derive(Debug, Default)]
pub struct GraphBuilder {
    resources: IndexMap<trace::VarId, (ResourceDesc, HashSet<trace::VarId>)>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn try_push_resource(
        &mut self,
        trace: &trace::Trace,
        id: trace::VarId,
    ) -> Option<ResourceId> {
        // Get the id of the variable holding the actual resource
        let rid = trace.resource_var(id);
        if let Some(rid) = self.resources.get_index_of(&rid) {
            self.resources[rid].1.insert(id);
            Some(ResourceId(rid))
        } else {
            let var = trace.var(rid);
            let desc = match var.op.resulting_op() {
                op::Op::Buffer => ResourceDesc::BufferDesc(BufferDesc {
                    size: var.extent.capacity(),
                    ty: var.ty,
                }),
                op::Op::Texture => {
                    let (shape, channels) = var.extent.shape_and_channles();

                    ResourceDesc::TextureDesc(TextureDesc {
                        shape,
                        channels,
                        format: var.ty,
                    })
                }
                op::Op::Accel => ResourceDesc::AccelDesc(var.extent.accel_desc().clone()),
                _ => return None,
            };
            Some(ResourceId(
                self.resources
                    .insert_full(rid, (desc, HashSet::from([(id)])))
                    .0,
            ))
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
    passes: Vec<Pass>,
    resource_descs: Vec<ResourceDesc>,
    resources: Vec<GraphResource>,
    // resource_vars: Vec<Vec<trace::VarId>>,
    n_outputs: usize,
}

impl Graph {
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }
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
    pub fn launch(&self, device: &backend::Device) -> Option<backend::Report> {
        Some(self.launch_with(device, &[])?.0)
    }
    ///
    /// This function launches the recorded graph with changed input variables.
    /// It should faile, if the input variables types or sizes changed.
    /// Therefore when recording a function, we keep multiple graphs with different input
    /// types/sizes.
    /// It might be possibel, to infer sizes on a compiled graph.
    /// Size independent kernels are cached in the backend layer anyways,
    /// therefore rerecording should be fine.
    ///
    #[profiling::function]
    pub fn launch_with(
        &self,
        device: &backend::Device,
        inputs: &[&trace::VarRef],
    ) -> Option<(backend::Report, Vec<trace::VarRef>)> {
        if self.passes.is_empty() {
            return None;
        }
        //
        // At first capture the current environment from referenced variables of the graph.
        //
        // TODO: at some point we could forward "virtual" resources to the backend, so that it can
        // perform resource aliasing
        let mut env = Env::default();
        trace::with_trace(|trace| {
            env.resources = self
                .resources
                .iter()
                .zip(self.resource_descs.iter())
                .map(|(res, desc)| match res {
                    GraphResource::Input { idx, .. } => {
                        let var = trace.var(inputs[*idx].id());
                        assert_eq!(var.resource_desc().as_ref(), Some(desc));
                        var.data.clone()
                    }
                    GraphResource::Output { .. } => Resource::create(device, desc),
                    GraphResource::Captured { r, .. } => trace.var(r.id()).data.clone(),
                    GraphResource::Internal { .. } => Resource::create(device, desc),
                })
                .collect::<Vec<_>>();
        });

        // Execute the graph with the captured environment on the device.
        log::trace!("Launching Graph");
        let report = device.execute_graph(self, &env).unwrap();
        log::trace!("Report:\n {report}");

        // Update variables from environment.
        // This does the following things:
        //      - Internal resources, that still exist are update
        //      - Variables for output resources are created
        //      - Create output variables (keeping in mind pass-through variables)

        let mut output = vec![None; self.n_outputs];
        env.resources
            .into_iter()
            .zip(self.resources.iter())
            .zip(self.resource_descs.iter())
            .for_each(|((res, gres), desc)| match gres {
                GraphResource::Internal { id, ids } => {
                    // Set resource (only if descriptors match) this way the resource desc of
                    // the variable never
                    // changes
                    // TODO: remove get_var_mut
                    trace::with_trace(|trace| {
                        if let Some(var) = trace.get_var_mut(*id) {
                            if let Some(var_resource_desc) = var.resource_desc() {
                                if &var_resource_desc == desc {
                                    for id in ids {
                                        trace.set_resource(*id, res.clone());
                                    }
                                }
                            }
                        }
                    });
                }
                GraphResource::Output { idx } => {
                    let op = res.op();
                    // Create a new variable for all output variables
                    let (ty, extent) = match desc {
                        ResourceDesc::BufferDesc(desc) => (desc.ty, Extent::Size(desc.size)),
                        ResourceDesc::TextureDesc(desc) => (
                            desc.format,
                            Extent::Texture {
                                shape: desc.shape,
                                channels: desc.channels,
                            },
                        ),
                        ResourceDesc::AccelDesc(desc) => {
                            (vartype::void(), Extent::Accel(desc.clone()))
                        }
                    };
                    trace::with_trace(|trace| {
                        output[*idx] = Some(trace.push_var(trace::Var {
                            op,
                            ty,
                            extent,
                            data: res,
                            ..Default::default()
                        }))
                    })
                }
                GraphResource::Input {
                    idx,
                    out_idx: Some(out_idx),
                } => {
                    // Use input variable for output
                    output[*out_idx] = Some(inputs[*idx].clone());
                }
                GraphResource::Captured { r, refs, out_idx } => trace::with_trace(|trace| {
                    for r in refs {
                        trace.set_resource(r.id(), res.clone());
                    }
                    if let Some(out_idx) = *out_idx {
                        output[out_idx] = Some(r.clone());
                    }
                }),
                _ => {}
            });

        let output = output.into_iter().collect::<Option<Vec<_>>>().unwrap();

        Some((report, output))
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

///
/// Compile the current thread state.
/// This results in a graph of passes, and resources.
///
/// * `ts`: The thread state to compile.
/// * `input`: Special variables, that are inputs to functions and might change between launches.
/// * `output`: Output variables, for which new variables have to be generated at every launch.
///
#[profiling::function]
pub fn compile(
    ts: trace::ThreadState,
    input: &[&trace::VarRef],
    output: &[&trace::VarRef],
) -> Graph {
    let mut graph_builder = GraphBuilder::default();

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

    fn groups_by_extent(
        trace: &mut trace::Trace,
        vars: &mut [trace::VarId],
    ) -> Vec<std::ops::Range<usize>> {
        vars.sort_by(|id0, id1| {
            trace
                .var(*id0)
                .extent
                .partial_cmp(&trace.var(*id1).extent)
                .unwrap()
        });

        let mut groups = vec![];
        let mut extent = Extent::default();
        let mut start = 0;
        for i in 0..vars.len() {
            let var = trace.var(vars[i]);
            if var.extent != extent || var.op.is_device_op() || matches!(var.op, op::Op::ScatterPhi)
            {
                let end = i + 1;
                if start != end {
                    groups.push(start..end);
                }
                extent = trace.var(vars[i]).extent.clone();
                start = end;
            }
        }
        groups
    }

    #[derive(Debug)]
    struct Var {
        id: trace::VarId,
        deps: Range<usize>,
        dc: usize,
    }

    fn topo_sort(
        trace: &trace::Trace,
        visited: &mut HashMap<trace::VarId, usize>,
        vars: &mut Vec<Var>,
        deps: &mut Vec<usize>,
        id: trace::VarId,
    ) -> usize {
        let i = if visited.contains_key(&id) {
            let i = visited[&id];
            i
        } else {
            let d = trace
                .var(id)
                .deps
                .iter()
                .map(|id| topo_sort(trace, visited, vars, deps, *id))
                .collect::<Vec<_>>();

            let start = deps.len();
            deps.extend(d.into_iter());
            let end = deps.len();

            let i = vars.len();
            vars.push(Var {
                id,
                deps: start..end,
                dc: 0,
            });
            visited.insert(id, i);
            i
        };
        // For all dependencies of this variable, increment its
        // dependant count
        for i in vars[i].deps.clone().map(|i| deps[i]) {
            vars[i].dc += 1;
        }
        i
    }

    let schedule = ts.scheduled.values().map(|r| r.id()).collect::<Vec<_>>();

    trace::with_trace(|trace| {
        let mut visited = HashMap::new();
        let mut deps: Vec<usize> = vec![];
        let mut vars: Vec<Var> = vec![];

        // Topo sort the whole graph starting at the schedule
        for id in &schedule {
            topo_sort(trace, &mut visited, &mut vars, &mut deps, *id);
        }

        let mut frontier = vars
            .iter()
            .filter(|var| var.dc == 0)
            .map(|var| var.id)
            .collect::<Vec<_>>();
        let mut next_frontier = vec![];

        // dbg!(&frontier);

        while !frontier.is_empty() {
            let groups = groups_by_extent(trace, &mut frontier);
            for group in groups {
                let first_id = frontier[group.start];
                let first_var = trace.var(first_id);

                // Note: we know, that all variables in a group have the same size
                // TODO: validate, that the the size_buffer is a buffer
                let size_buffer = match first_var.extent {
                    Extent::DynSize { size: size_dep, .. } => {
                        graph_builder.try_push_resource(trace, size_dep)
                    }
                    _ => None,
                };

                let deps = if first_var.op.is_device_op() {
                    // Handle Device Ops (precompiled)
                    assert_eq!(group.len(), 1);
                    let id = frontier[group.start];

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
                                size_buffer,
                                op: PassOp::DeviceOp(op),
                            });
                            deps
                        }
                        _ => todo!(),
                    }
                } else if matches!(first_var.op, op::Op::ScatterPhi) {
                    assert_eq!(group.len(), 1);
                    graph_builder.try_push_resource(trace, first_id);
                    let var = trace.var(frontier[group.start]);
                    var.deps.clone()
                } else {
                    let mut compiler = compiler::Compiler::default();

                    let deps = compiler.compile(trace, &frontier[group.clone()]);

                    let resources = [
                        compiler.buffers.iter(),
                        compiler.textures.iter(),
                        compiler.accels.iter(),
                    ]
                    .into_iter()
                    .flatten()
                    .flat_map(|&id| graph_builder.try_push_resource(trace, id))
                    .collect::<Vec<_>>();

                    let ir = std::mem::take(&mut compiler.ir);
                    let size = trace.var(frontier[group.start]).extent.capacity();

                    graph_builder.push_pass(Pass {
                        resources,
                        size_buffer,
                        op: PassOp::Kernel { ir, size },
                    });
                    deps
                };

                // Add dependencies to next frontier
                // If their dependant count is 0
                // otherwise decrement it.
                for i in deps.iter().map(|id| visited[id]) {
                    vars[i].dc -= 1;
                }
                next_frontier.extend(deps.into_iter().map(|id| visited[&id]).filter_map(|i| {
                    if vars[i].dc == 0 {
                        Some(vars[i].id)
                    } else {
                        None
                    }
                }));
                dbg!(&next_frontier);

                // graph_builder.push_pass(pass);
                // Put the variables in this group into their evaluated state, removing dependencies and
                // changing the op type.
                // for i in group.clone() {
                //     let id = frontier[i];
                //     trace.advance(id);
                // }
            }
            std::mem::swap(&mut frontier, &mut next_frontier);
            next_frontier.clear();
        }

        // // Advance all resource variables to the state after evaluation
        // for (&id, _) in graph_builder.resources.iter() {
        //     trace.advance(id);
        // }
        for (_, (_, ids)) in graph_builder.resources.iter() {
            for &id in ids {
                trace.advance(id);
            }
        }

        let resources = graph_builder
            .resources
            .iter()
            .map(|(id, (_, ids))| {
                if input.contains_key(&id) {
                    GraphResource::Input {
                        idx: input[&id],
                        out_idx: output.get(id).cloned(),
                    }
                } else if matches!(
                    trace.var(*id).data,
                    Resource::Buffer(_) | Resource::Texture(_) | Resource::Accel(_)
                ) {
                    GraphResource::Captured {
                        r: trace.ref_borrow(*id),
                        refs: ids.iter().map(|id| trace.ref_borrow(*id)).collect(),
                        out_idx: output.get(id).cloned(),
                    }
                } else if output.contains_key(&id) {
                    GraphResource::Output { idx: output[&id] }
                } else {
                    GraphResource::Internal {
                        id: *id,
                        ids: ids.iter().cloned().collect(),
                    }
                }
            })
            .collect::<Vec<_>>();

        let resource_descs = graph_builder
            .resources
            .into_iter()
            .map(|(_, (desc, _))| desc)
            .collect::<Vec<_>>();

        // Reverse passes, as they have been generated in reverse
        graph_builder.passes.reverse();

        let graph = Graph {
            passes: graph_builder.passes,
            resource_descs,
            resources,
            n_outputs,
        };
        graph
    })
}
