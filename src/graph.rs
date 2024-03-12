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

// #[derive(Debug)]
// pub enum GraphResource {
//     Internal,
// }

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
        with_ftrace(|ftrace| {
            let mut resources: IndexMap<Var, ()> = IndexMap::default();
            let mut passes = vec![];

            let mut push_resource = |ftrace: &mut FTrace, var: Var| {
                if let Some(id) = resources.get_index_of(&var) {
                    Some(ResourceId(id))
                } else {
                    let entry = ftrace.entry(var);
                    let id = resources.insert_full(var, ()).0;
                    Some(ResourceId(id))
                }
            };

            // Add input/output resources
            let input = input
                .into_iter()
                .map(|var| push_resource(ftrace, *var))
                .collect::<Option<Vec<_>>>()
                .unwrap();
            let output = output
                .into_iter()
                .map(|var| push_resource(ftrace, *var))
                .collect::<Option<Vec<_>>>()
                .unwrap();

            let groups = ftrace.groups.clone();
            let mut schedule = ftrace.scheduled.clone();

            // Subdivide groups by extent
            let groups = groups
                .iter()
                .flat_map(|group| {
                    schedule[group.clone()].sort_by(|v0, v1| {
                        ftrace
                            .entry(*v0)
                            .extent
                            .partial_cmp(&ftrace.entry(*v1).extent)
                            .unwrap()
                    });

                    let mut groups = vec![];
                    let mut size = Extent::default();
                    let mut start = group.start;

                    for i in group.clone() {
                        if ftrace.entry(schedule[i]).extent != size {
                            let end = i + 1;
                            if start != end {
                                groups.push(start..end);
                            }
                            size = ftrace.entry(schedule[i]).extent.clone();
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

            // Compile the graph
            for group in groups.iter() {
                let var = schedule[group.start];
                let entry = ftrace.entry(var);

                let size_buffer = None;

                let pass = if entry.op.is_device_op() {
                    assert_eq!(group.len(), 1);

                    match entry.op {
                        Op::DeviceOp(op) => {
                            let deps = &entry.deps.clone();

                            // TODO: Improve the readability here. atm. we are pushing all the
                            // dependenceis into multiple vecs starting with [id]
                            let resources = [var]
                                .iter()
                                .chain(deps.iter())
                                .flat_map(|var| push_resource(ftrace, *var))
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
                    let mut compiler = compiler::Compiler::default();

                    compiler.compile(ftrace, &schedule[group.clone()]);

                    let resources = compiler
                        .buffers
                        .iter()
                        .chain(compiler.textures.iter())
                        .chain(compiler.accels.iter())
                        .flat_map(|&var| push_resource(ftrace, var))
                        .collect::<Vec<_>>();

                    let ir = compiler.ir;

                    let size = ftrace.entry(schedule[group.start]).extent.capacity();
                    let ir = Prehashed::new(ir);

                    Pass {
                        resources,
                        size_buffer,
                        op: PassOp::Kernel { ir, size },
                    }
                };
                passes.push(pass);
                for i in group.clone() {
                    let var = schedule[i];
                    // TODO: advance
                    let entry = ftrace.entry_mut(var);
                    entry.op = entry.op.resulting_op();
                }
            }
            let resources = resources
                .into_keys()
                .map(|var| ftrace.entry(var).desc())
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
