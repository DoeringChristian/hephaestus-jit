use std::cell::RefCell;
use std::thread::ThreadId;

use crate::backend::Device;
use crate::data::Data;
use crate::op::Op;
use crate::vartype::VarType;
use crate::{compiler, graph};
use slotmap::{DefaultKey, SlotMap};

thread_local! {
    pub static TRACE: RefCell<Trace> = RefCell::new(Default::default());
    pub static SCHEDULE: RefCell<Vec<VarRef>> = RefCell::new(Default::default());
}

#[derive(Default, Debug)]
pub struct Trace {
    vars: SlotMap<DefaultKey, Var>,
    pub device: Option<Device>,
}
impl Trace {
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }
    pub fn get_var(&mut self, id: VarId) -> Option<&Var> {
        self.vars.get(id.0)
    }
    pub fn push_var(&mut self, mut v: Var) -> VarId {
        for dep in v.deps.iter() {
            self.inc_rc(*dep);
        }
        v.rc = 1;
        let id = VarId(self.vars.insert(v));
        id
    }
    pub fn inc_rc(&mut self, id: VarId) {
        self.var_mut(id).rc += 1;
    }
    pub fn dec_rc(&mut self, id: VarId) {
        let var = self.var_mut(id);
        var.rc -= 1;
        if var.rc == 0 {
            for dep in var.deps.clone() {
                self.dec_rc(dep);
            }
            let var = self.var_mut(id);
            self.vars.remove(id.0);
        }
    }
    pub fn var_info(&self, ids: &[VarId]) -> VarInfo {
        let ty = self.var(*ids.first().unwrap()).ty.clone(); // TODO: Fix (first non void)

        let size = ids
            .iter()
            .map(|id| self.var(*id).size)
            .reduce(|s0, s1| s0.max(s1))
            .unwrap()
            .clone();
        VarInfo { ty, size }
    }
    pub fn ref_borrow(&mut self, id: VarId) -> VarRef {
        self.inc_rc(id);
        VarRef {
            id,
            _thread_id: std::thread::current().id(),
        }
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        assert_eq!(self.vars.len(), 0, "{self:#?}");
    }
}

#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(DefaultKey);

#[derive(Debug)]
pub struct VarRef {
    id: VarId,
    _thread_id: ThreadId,
}

impl VarRef {
    pub fn id(&self) -> VarId {
        self.id
    }
}

impl Clone for VarRef {
    fn clone(&self) -> Self {
        with_trace(|t| t.inc_rc(self.id()));
        Self {
            id: self.id,
            _thread_id: self._thread_id,
        }
    }
}
impl Drop for VarRef {
    fn drop(&mut self) {
        with_trace(|t| t.dec_rc(self.id()))
    }
}

#[derive(Debug, Default)]
pub struct Var {
    pub op: Op, // Operation used to construct the variable
    pub deps: Vec<VarId>,
    pub ty: VarType, // Type of the variable
    pub size: usize, // number of elements
    pub rc: usize,

    // pub arrays: Option<Array>,
    pub data: Data,
}
#[derive(Debug)]
pub struct VarInfo {
    pub ty: VarType,
    pub size: usize,
}

pub fn with_trace<T, F: FnOnce(&mut Trace) -> T>(f: F) -> T {
    TRACE.with(|t| {
        let mut t = t.borrow_mut();
        f(&mut t)
    })
}
fn push_var(v: Var) -> VarRef {
    with_trace(|t| VarRef {
        id: t.push_var(v),
        _thread_id: std::thread::current().id(),
    })
}

pub fn eval(refs: &[&VarRef]) {
    with_trace(|t| {
        compiler::eval(t, refs.iter().map(|r| r.id()));
    })
}
pub fn compile() -> graph::Graph {
    SCHEDULE.with(|s| {
        let mut s = s.borrow_mut();
        let schedule = s.iter().map(|r| r.id).collect::<Vec<_>>();
        let graph = with_trace(|t| graph::compile(t, schedule));
        s.clear();
        graph
    })
}

// Trace Functions
pub fn index(size: usize) -> VarRef {
    push_var(Var {
        op: Op::Index,
        deps: vec![],
        ty: VarType::U32,
        size,
        ..Default::default()
    })
}

impl VarRef {
    pub fn add(&self, other: &VarRef) -> VarRef {
        assert_eq!(self._thread_id, std::thread::current().id());
        assert_eq!(other._thread_id, std::thread::current().id());
        let info = with_trace(|t| t.var_info(&[self.id(), other.id()]));
        push_var(Var {
            op: Op::Add,
            deps: vec![self.id(), other.id()],
            ty: info.ty,
            size: info.size,
            ..Default::default()
        })
    }
    pub fn data(&self) -> Data {
        assert_eq!(self._thread_id, std::thread::current().id());
        with_trace(|t| t.var(self.id()).data.clone())
    }
    pub fn same_trace(&self, other: &VarRef) -> bool {
        self._thread_id == other._thread_id
    }
    pub fn schedule(&self) {
        SCHEDULE.with(|s| {
            let mut s = s.borrow_mut();
            s.push(self.clone());
        })
    }
    pub fn rc(&self) -> usize {
        with_trace(|t| t.var(self.id()).rc)
    }
}
