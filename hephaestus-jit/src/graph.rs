use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::{Div, Range};

use crate::extent::Extent;
use crate::prehashed::Prehashed;
use crate::resource::{BufferDesc, Resource, ResourceDesc, TextureDesc};
use crate::{backend, vartype};
use crate::{compiler, ir, op, trace};
use indexmap::IndexMap;

use num_traits::Float;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to add a resource to the graph!")]
    PushResourceError(Backtrace),
    #[error("Resource does not match variable type!")]
    ResourceMissmatch,
    #[error("A resource in the environment has been left empty!")]
    UninitializedResourve,
    #[error("Could not compare extent!")]
    ExtentComparison,
    #[error(transparent)]
    BackendError(#[from] backend::Error),
    #[error("Undefined error!")]
    None(Backtrace),
}

pub type Result<T> = std::result::Result<T, Error>;

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
    // An input variable to the Graph
    Input,
    // Using strong reference for captured variables
    Captured { r: trace::VarRef },
    // Using weak reference for internal variables
    Internal { id: trace::VarId },
}
// Have to implement debug for snapshot tests to work
impl Debug for GraphResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => f.debug_struct("Input").finish(),
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
    resources: IndexMap<trace::VarId, ResourceDesc>,
    passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn try_push_resource(
        &mut self,
        trace: &mut trace::Trace,
        id: trace::VarId,
    ) -> Result<ResourceId> {
        if let Some(id) = self.resources.get_index_of(&id) {
            Ok(ResourceId(id))
        } else {
            let var = trace.var(id);
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
                _ => return Err(Error::ResourceMissmatch),
            };
            //NOTE: we increment the rc of the variable, to prevent premature dropping of resources
            //by the `advance` function
            trace.inc_rc(id);
            Ok(ResourceId(self.resources.insert_full(id, desc).0))
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
pub struct Report {
    pub backend: backend::Report,
    pub aliasing_rate: f32,
}
impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Backend:\n {}", self.backend)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct Graph {
    passes: Vec<Pass>,
    resource_descs: Vec<ResourceDesc>,
    resources: Vec<GraphResource>,
    inputs: Vec<ResourceId>,
    outputs: Vec<ResourceId>,
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
    pub fn launch(&self, device: &backend::Device) -> Result<Report> {
        Ok(self.launch_with(device, &[])?.0)
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
    pub fn launch_with(
        &self,
        device: &backend::Device,
        inputs: &[trace::VarRef],
    ) -> Result<(Report, Vec<trace::VarRef>)> {
        log::trace!("Launching graph: {self:?}");
        //
        // At first capture the current environment from referenced variables of the graph.
        //
        // TODO: at some point we could forward "virtual" resources to the backend, so that it can
        // perform resource aliasing
        let mut resources = vec![None; self.resources.len()];

        for (r, id) in inputs.iter().zip(self.inputs.iter()) {
            trace::with_trace(|trace| -> Result<()> {
                let var = trace.var(r.id());
                let desc = &self.resource_descs[id.0];

                // Assert
                (var.resource_desc().as_ref() == Some(desc))
                    .then_some(())
                    .ok_or(Error::ResourceMissmatch)?;

                resources[id.0] = Some(var.data.clone());

                Ok(())
            })?;
        }
        for i in 0..resources.len() {
            if resources[i].is_none() {
                match &self.resources[i] {
                    GraphResource::Captured { r } => trace::with_trace(|trace| {
                        resources[i] = Some(trace.var(r.id()).data.clone());
                    }),
                    GraphResource::Internal { id }
                        if trace::with_trace(|trace| trace.get_var(*id).is_some()) =>
                    {
                        resources[i] = Some(Resource::create(device, &self.resource_descs[i]))
                        // TODO: error handling
                    }
                    _ => {}
                }
            }
        }

        // Calculate resource aliasing for internal resources
        // NOTE: aliasing in the backend might be more efficient.

        // Livetimes of resources (inclusive range)
        let mut livetimes: Vec<_> = vec![(resources.len(), 0); resources.len()];
        for (pid, pass) in self.passes.iter().enumerate() {
            for &rid in &pass.resources {
                let livetime = &mut livetimes[rid.0];
                *livetime = (livetime.0.min(pid), livetime.1.max(pid))
            }
        }

        let mut n_internal = 0usize;
        let internal = resources
            .iter()
            .map(|r| {
                if r.is_none() {
                    n_internal += 1;
                    true
                } else {
                    false
                }
            })
            .collect::<Vec<_>>();

        // Calculate aliasing using cache and linear sweep
        let mut misses = 0usize;
        let mut cache: HashMap<ResourceDesc, Vec<_>> = HashMap::new();
        for (pid, pass) in self.passes.iter().enumerate() {
            for rid in &pass.resources {
                let livetime = livetimes[rid.0];
                if internal[rid.0] {
                    if livetime.0 == pid {
                        assert!(resources[rid.0].is_none());
                        // Allocate or reuse resource at the beginning of it's livetime
                        let desc = self.resource_descs[rid.0].round_up();

                        let resource = cache
                            .entry(desc.clone())
                            .or_insert_with(|| vec![])
                            .pop()
                            .unwrap_or_else(|| {
                                misses += 1;
                                Resource::create(device, &desc)
                            });
                        resources[rid.0] = Some(resource);
                    }
                    if livetime.1 == pid {
                        // Deallocate / Reinsert into cache
                        let desc = self.resource_descs[rid.0].round_up();
                        cache
                            .get_mut(&desc)
                            .unwrap()
                            .push(resources[rid.0].as_ref().unwrap().clone());
                    }
                }
            }
        }

        let hit_rate = {
            let hit_rate = (1.0 - (misses as f32).div(n_internal as f32));
            if hit_rate.is_nan() {
                0.
            } else {
                hit_rate
            }
        };
        log::trace!(
            "Calculated aliasing: {n_internal} internal resources, {misses} cache misses, {hit_rate}% hit rate", hit_rate = hit_rate*100.
        );

        let resources = resources
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::UninitializedResourve)?;

        let env = Env { resources };

        // Execute the graph with the captured environment on the device.
        log::trace!("Launching Graph");
        let backend_report = if self.passes.is_empty() {
            backend::Report::default()
        } else {
            device.execute_graph(self, &env)?
        };
        log::trace!("Backend:\n {backend_report}");

        // Update variables from environment.
        // This does the following things:
        //      - Internal resources, that still exist are update
        //      - Variables for output resources are created
        //      - Create output variables (keeping in mind pass-through variables)

        let output = self
            .outputs
            .iter()
            .map(|id| {
                let res = env.resources[id.0].clone();
                let desc = &self.resource_descs[id.0];

                let op = match res {
                    Resource::Buffer(_) => op::Op::Buffer,
                    Resource::Texture(_) => op::Op::Texture,
                    Resource::Accel(_) => op::Op::Accel,
                    _ => todo!(),
                };
                let (ty, extent) = match desc {
                    ResourceDesc::BufferDesc(desc) => (desc.ty, Extent::Size(desc.size)),
                    ResourceDesc::TextureDesc(desc) => (
                        desc.format,
                        Extent::Texture {
                            shape: desc.shape,
                            channels: desc.channels,
                        },
                    ),
                    ResourceDesc::AccelDesc(desc) => (vartype::void(), Extent::Accel(desc.clone())),
                };
                trace::with_trace(|trace| {
                    trace.new_var(trace::Var {
                        op,
                        ty,
                        extent,
                        data: res,
                        ..Default::default()
                    })
                })
            })
            .collect::<Vec<_>>();

        env.resources
            .into_iter()
            .zip(self.resources.iter())
            .zip(self.resource_descs.iter())
            .for_each(|((res, gres), desc)| match gres {
                GraphResource::Internal { id } => {
                    // Set resource (only if descriptors match) this way the resource desc of
                    // the variable never
                    // changes
                    // TODO: remove get_var_mut
                    trace::with_trace(|trace| {
                        if let Some(var) = trace.get_var_mut(*id) {
                            if let Some(var_resource_desc) = var.resource_desc() {
                                if &var_resource_desc == desc {
                                    var.data = res;
                                }
                            }
                        }
                    });
                }
                _ => {}
            });

        profiling::finish_frame!();
        let report = Report {
            backend: backend_report,
            aliasing_rate: hit_rate,
        };
        Ok((report, output))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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
        ir: Prehashed<ir::IR>,
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
    ts: &trace::ThreadState,
    input: &[trace::VarRef],
    output: &[trace::VarRef],
) -> Result<Graph> {
    // TODO: Not sure if we should lock the graph for the whole compilation?
    trace::with_trace(|trace| {
        let mut graph_builder = GraphBuilder::default();
        for r in input.iter().chain(output.iter()) {
            graph_builder.try_push_resource(trace, r.id())?;
        }

        let inputs = input
            .iter()
            .map(|r| graph_builder.try_push_resource(trace, r.id()))
            .collect::<Result<Vec<_>>>()?;
        let outputs = output
            .iter()
            .map(|r| graph_builder.try_push_resource(trace, r.id()))
            .collect::<Result<Vec<_>>>()?;
        let input_set = inputs.iter().copied().collect::<HashSet<_>>();

        // Get scheduled variables from thread state in order
        let mut vars = ts.scheduled.values().map(|r| r.id()).collect::<Vec<_>>();
        let groups = ts.groups.clone();

        // Subdivide groups by extent
        let groups = groups
            .iter()
            .flat_map(|group| {
                vars[group.clone()].sort_by(|id0, id1| {
                    // TODO: Result handling
                    trace
                        .var(*id0)
                        .extent
                        .partial_cmp(&trace.var(*id1).extent)
                        .unwrap()
                });

                let mut groups = vec![];
                let mut size = trace.var(vars[group.start]).extent.clone();
                let mut start = group.start;

                for i in group.clone() {
                    if trace.var(vars[i]).extent != size {
                        let end = i;
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
        for group in groups.iter() {
            let first_id = vars[group.start];
            let extent = trace.var(first_id).extent.clone();
            // Note: we know, that all variables in a group have the same size
            // TODO: validate, that the the size_buffer is a buffer
            let size_buffer = match extent {
                Extent::DynSize { size: size_dep, .. } => {
                    graph_builder.try_push_resource(trace, size_dep).ok()
                }
                _ => None,
            };
            let first_var = trace.var(first_id);

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
                        .resource_vars()
                        .flat_map(|id| graph_builder.try_push_resource(trace, id))
                        .collect::<Vec<_>>();

                    let ir = std::mem::take(&mut compiler.ir);
                    (resources, ir)
                });
                let size = trace.var(vars[group.start]).extent.capacity();
                let ir = Prehashed::new(ir);
                Pass {
                    resources,
                    size_buffer,
                    op: PassOp::Kernel { ir, size },
                }
            };

            graph_builder.push_pass(pass);
            // Put the variables in this group into their evaluated state, removing dependencies and
            // changing the op type.
            for i in group.clone() {
                let id = vars[i];
                trace.advance(id);
            }
        }

        // Collect descriptors and input resources
        let resources = graph_builder
            .resources
            .keys()
            .enumerate()
            .map(|(i, var_id)| {
                let resource_id = ResourceId(i);
                let res = if input_set.contains(&resource_id) {
                    GraphResource::Input
                } else if matches!(
                    trace.var(*var_id).data,
                    Resource::Buffer(_) | Resource::Texture(_) | Resource::Accel(_)
                ) {
                    GraphResource::Captured {
                        r: trace.ref_borrow(*var_id),
                    }
                } else {
                    GraphResource::Internal { id: *var_id }
                };
                // NOTE: decrement rc of resource again
                trace.dec_rc(*var_id);
                res
            })
            .collect::<Vec<_>>();

        let resource_descs = graph_builder
            .resources
            .into_iter()
            .map(|(_, desc)| desc)
            .collect::<Vec<_>>();

        let graph = Graph {
            passes: graph_builder.passes,
            resource_descs,
            resources,
            inputs,
            outputs,
        };
        Ok(graph)
    })
}
