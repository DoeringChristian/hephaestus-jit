use std::cell::RefCell;
use std::ops::Range;

use indexmap::IndexMap;

use crate::extent::Extent;
use crate::ir::IR;
use crate::op::{DeviceOp, KernelOp, Op};
use crate::prehashed::Prehashed;
use crate::vartype::{self, VarType};
use crate::{compiler, AsVarType};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Var(usize);

#[derive(Debug)]
pub struct Entry {
    pub op: Op,
    pub ty: &'static VarType,
    pub extent: Extent,
    pub deps: Vec<Var>,
    pub literal: u64,
}
impl Default for Entry {
    fn default() -> Self {
        Self {
            op: Default::default(),
            ty: vartype::void(),
            extent: Default::default(),
            deps: Default::default(),
            literal: Default::default(),
        }
    }
}

#[derive(Default, Debug)]
pub struct FTrace {
    entries: Vec<Entry>,
    scheduled: Vec<Var>,
    pub groups: Vec<Range<usize>>,
}

impl FTrace {
    pub fn new_var(&mut self, entry: Entry) -> Var {
        let i = self.entries.len();
        self.entries.push(entry);
        Var(i)
    }
    pub fn entry(&self, var: Var) -> &Entry {
        &self.entries[var.0]
    }
}

thread_local! {
    pub static FTRACE: RefCell<FTrace> = RefCell::new(Default::default());
}

pub fn new_var(entry: Entry) -> Var {
    FTRACE.with(|ftrace| {
        let mut ftrace = ftrace.borrow_mut();
        ftrace.new_var(entry)
    })
}

pub fn with_ftrace<T, F: FnOnce(&mut FTrace) -> T>(f: F) -> T {
    FTRACE.with(|t| {
        let mut ftrace = t.borrow_mut();
        f(&mut ftrace)
    })
}

///
/// Returns a variable representing a literal within a kernel.
/// In contrast to [sized_literal], it cannot be evaluated.
///
pub fn literal<T: AsVarType>(val: T) -> Var {
    let ty = T::var_ty();
    let mut data = 0;
    unsafe { *(&mut data as *mut _ as *mut T) = val };
    new_var(Entry {
        op: Op::KernelOp(KernelOp::Literal),
        ty,
        extent: Extent::None,
        literal: data,
        ..Default::default()
    })
}
///
/// Returns a variable representing a literal within a kernel.
/// This operation also has an inpact on the size of variables depending on it.
/// It might be used to initialize buffers for device operations.
///
pub fn sized_literal<T: AsVarType>(val: T, size: usize) -> Var {
    let ty = T::var_ty();
    let mut data = 0;
    unsafe { *(&mut data as *mut _ as *mut T) = val };
    new_var(Entry {
        op: Op::KernelOp(KernelOp::Literal),
        ty,
        extent: Extent::Size(size),
        literal: data,
        ..Default::default()
    })
}

impl Var {
    pub fn schedule(&self) {
        with_ftrace(|ftrace| {
            ftrace.scheduled.push(*self);
        });
    }
}

// Graph

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
    resources: Vec<Var>,
}

#[derive(Debug)]
pub enum GraphResource {
    Internal { var: Var },
}

impl Graph {
    pub fn compile() -> Self {
        with_ftrace(|ftrace| {
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

            let mut resources: IndexMap<Var, GraphResource> = IndexMap::default();
            let mut push_resource = |var: Var|{
                if let Some(id) = resources.get_index_of(&var){
                    Some(ResourceId(id))
                }else{
                    let entry = ftrace.entry(var);
                    let id = resources.insert_full(var, GraphResource::Internal{var}).0;
                    Some(ResourceId(id))
                }
            };

            // Compile the graph
            for group in groups.iter() {
                let var = schedule[group.start];
                let entry = ftrace.entry(var);

                    let size_buffer = None;

                let pass = if entry.op.is_device_op(){
                    assert_eq!(group.len(), 1);

                    match entry.op {
                        Op::DeviceOp(op) => {
                            let deps = &entry.deps;

                            // TODO: Improve the readability here. atm. we are pushing all the
                            // dependenceis into multiple vecs starting with [id]
                            let resources = [var]
                                .iter()
                                .chain(deps.iter())
                                .flat_map(|var| push_resource(*var))
                                .collect::<Vec<_>>();

                            Pass {
                                resources,
                                size_buffer,
                                op: PassOp::DeviceOp(op),
                            }
                        }
                        _ => todo!(),
                    }
                }else{
                    let mut compiler = compiler::Compiler::default();

                    compiler.compile(ftrace, &schedule[group.clone()]);

                    let resources = compiler
                        .buffers
                        .iter()
                        .chain(compiler.textures.iter())
                        .chain(compiler.accels.iter())
                        .flat_map(|&var| push_resource(var))
                        .collect::<Vec<_>>();

                    let ir = compiler.ir;

                    let size = ftrace.entry(schedule[group.start]).extent.capacity();
                    let ir = Prehashed::new(ir);

                    Pass{
                        resources,
                        size_buffer,
                        op: PassOp::Kernel{ir, size}
                    };

                        
                    
                    todo!()
                }
                todo!();
            }
        });
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn ftrace_01() {
        let x = sized_literal(1, 10);
        x.schedule();
    }
}
