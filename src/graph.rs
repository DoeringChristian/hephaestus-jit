// Graph

use indexmap::IndexMap;

use crate::backend::{AccelDesc, ArrayDesc};
use crate::extent::Extent;
use crate::ftrace::{with_ftrace, FTrace, Var};
use crate::ir::IR;
use crate::op::{DeviceOp, Op};
use crate::prehashed::Prehashed;
use crate::resource::{Resource, ResourceDesc};
use crate::{backend, compiler};

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
        ir: Prehashed<IR>,
        size: usize,
    },
    DeviceOp(DeviceOp),
}

#[derive(Debug)]
pub struct Graph {
    passes: Vec<Pass>,
    resources: Vec<ResourceDesc>,
    input: Vec<ResourceId>,
    output: Vec<ResourceId>,
}

#[derive(Debug, Default, Clone)]
pub struct Env {
    resources: Vec<Option<Resource>>,
}
impl Env {
    pub fn buffer(&self, id: ResourceId) -> Option<&backend::Array> {
        match &self.resources[id.0] {
            Some(Resource::Array(buffer)) => Some(buffer),
            _ => None,
        }
    }
    pub fn texture(&self, id: ResourceId) -> Option<&backend::Texture> {
        match &self.resources[id.0] {
            Some(Resource::Texture(texture)) => Some(texture),
            _ => None,
        }
    }
    pub fn accel(&self, id: ResourceId) -> Option<&backend::Accel> {
        match &self.resources[id.0] {
            Some(Resource::Accel(accel)) => Some(accel),
            _ => None,
        }
    }
}

impl Graph {
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }
    pub fn array_desc(&self, id: ResourceId) -> ArrayDesc {
        match &self.resources[id.0] {
            ResourceDesc::ArrayDesc(desc) => desc.clone(),
            _ => todo!(),
        }
    }
    pub fn accel_desc(&self, id: ResourceId) -> AccelDesc {
        match &self.resources[id.0] {
            ResourceDesc::AccelDesc(desc) => desc.clone(),
            _ => todo!(),
        }
    }
    pub fn launch(&self, device: &backend::Device, input: &[Resource]) -> Vec<Resource> {
        let mut env = Env {
            resources: vec![None; self.resources.len()],
        };
        for (i, resource) in input.into_iter().enumerate() {
            env.resources[self.input[i].0] = Some(resource.clone());
        }
        for (i, desc) in self.resources.iter().enumerate() {
            if env.resources[i].is_none() {
                env.resources[i] = Some(Resource::create(device, desc));
            }
        }

        // Launch graph on device

        device.execute_graph(&self, &env).unwrap();

        let output = self
            .output
            .iter()
            .map(|id| env.resources[id.0].clone())
            .collect::<Option<Vec<_>>>();

        output.unwrap()
    }
    pub fn compile(input: &[Var], output: &[Var]) -> Self {
        with_ftrace(|trace| {
            let mut resources: IndexMap<Var, ()> = IndexMap::default();
            let mut passes = vec![];

            let schedule = output;

            let mut push_resource = |trace: &mut FTrace, var: Var| {
                // Get the first resource var
                fn resource_var(trace: &FTrace, var: Var) -> Var {
                    let entry = trace.entry(var);
                    if matches!(entry.op, Op::Buffer | Op::Texture | Op::Accel) {
                        if entry.deps.len() > 0 {
                            return entry.deps[0];
                        }
                    }
                    var
                }
                let var = resource_var(trace, var);

                if let Some(id) = resources.get_index_of(&var) {
                    Some(ResourceId(id))
                } else {
                    let id = resources.insert_full(var, ()).0;
                    Some(ResourceId(id))
                }
            };

            // Add input/output resources
            let input = input
                .into_iter()
                .map(|var| push_resource(trace, *var))
                .collect::<Option<Vec<_>>>()
                .unwrap();
            let output = output
                .into_iter()
                .map(|var| push_resource(trace, *var))
                .collect::<Option<Vec<_>>>()
                .unwrap();

            fn groups_by_extent(
                trace: &mut FTrace,
                vars: &mut [Var],
            ) -> Vec<std::ops::Range<usize>> {
                vars.sort_by(|id0, id1| {
                    trace
                        .entry(*id0)
                        .extent
                        .partial_cmp(&trace.entry(*id1).extent)
                        .unwrap_or(std::cmp::Ordering::Less)
                });

                let mut groups = vec![];
                let mut extent = Extent::default();
                let mut start = 0;
                for i in 0..vars.len() {
                    let var = trace.entry(vars[i]);
                    if var.extent != extent
                        || var.op.is_device_op()
                        || matches!(var.op, Op::Buffer | Op::Texture | Op::Accel)
                    {
                        let end = i + 1;
                        if start != end {
                            groups.push(start..end);
                        }
                        extent = trace.entry(vars[i]).extent.clone();
                        start = end;
                    }
                }
                groups
            }

            let mut depcount = vec![0; trace.entries.len()];

            for entry in &trace.entries {
                for dep in &entry.deps {
                    depcount[dep.0] += 1;
                }
            }

            let mut frontier = schedule.to_vec();
            let mut next_frontier = vec![];

            while !frontier.is_empty() {
                let groups = groups_by_extent(trace, &mut frontier);
                for group in groups {
                    let first_var = frontier[group.start];
                    // let entry = trace.entry(var);

                    // let size_buffer = match first_var.extent {
                    //     Extent::DynSize { size: size_dep, .. } => {
                    //         graph_builder.try_push_resource(trace, size_dep)
                    //     }
                    //     _ => None,
                    // };
                    let size_buffer = None;

                    let deps = if trace.entry(first_var).op.is_device_op() {
                        let entry = trace.entry(first_var);
                        assert_eq!(group.len(), 1);
                        let id = frontier[group.start];

                        match trace.entry(first_var).op {
                            Op::DeviceOp(op) => {
                                let deps = entry.deps.clone();

                                // TODO: Improve the readability here. atm. we are pushing all the
                                // dependenceis into multiple vecs starting with [id]
                                let resources = [id]
                                    .iter()
                                    .chain(deps.iter())
                                    .flat_map(|id| push_resource(trace, *id))
                                    .collect::<Vec<_>>();

                                passes.push(Pass {
                                    resources,
                                    size_buffer,
                                    op: PassOp::DeviceOp(op),
                                });
                                deps
                            }
                            _ => todo!(),
                        }
                    } else if matches!(
                        trace.entry(first_var).op,
                        Op::Buffer | Op::Texture | Op::Accel
                    ) {
                        assert_eq!(group.len(), 1);
                        push_resource(trace, first_var);
                        trace.entry(first_var).deps.clone()
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
                        .flat_map(|&id| push_resource(trace, id))
                        .collect::<Vec<_>>();

                        let ir = Prehashed::new(compiler.ir);
                        let size = trace.entry(first_var).extent.capacity();

                        passes.push(Pass {
                            resources,
                            size_buffer,
                            op: PassOp::Kernel { ir, size },
                        });
                        deps
                    };
                    // Add dependencies to next frontier
                    // If their dependant count is 0
                    // otherwise decrement it.

                    for dep in &deps {
                        depcount[dep.0] -= 1;
                    }

                    next_frontier.extend(deps.into_iter().filter_map(|var| {
                        if depcount[var.0] == 0 {
                            Some(var)
                        } else {
                            None
                        }
                    }));
                }
                // Swap out frontier with next_frontier
                std::mem::swap(&mut frontier, &mut next_frontier);
                next_frontier.clear();
            }

            // We have to reverse the passes, since we generate them in reverse order
            passes.reverse();

            let resources = resources
                .into_keys()
                .map(|var| trace.entry(var).desc())
                .collect::<Vec<_>>();
            Graph {
                passes,
                resources,
                input,
                output,
            }
        })
    }
}
