use std::cell::RefCell;
use std::thread::ThreadId;

use crate::data::Data;
use crate::ir;
use crate::op::{DeviceOp, Op};
use crate::vartype::{AsVarType, VarType};
use crate::{compiler, graph};
use slotmap::{DefaultKey, SlotMap};

thread_local! {
    pub static TRACE: RefCell<Trace> = RefCell::new(Default::default());
    pub static SCHEDULE: RefCell<Vec<VarRef>> = RefCell::new(Default::default());
}

#[derive(Default, Debug)]
pub struct Trace {
    vars: SlotMap<DefaultKey, Var>,
    // pub device: Option<Device>,
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

// pub fn eval(refs: &[&VarRef]) {
//     with_trace(|t| {
//         compiler::eval(t, refs.iter().map(|r| r.id()));
//     })
// }
pub fn compile() -> graph::Graph {
    SCHEDULE.with(|s| {
        let mut s = s.borrow_mut();
        let schedule = std::mem::take(s.as_mut());
        let graph = with_trace(|t| graph::compile(t, schedule));
        graph
    })
}

// Trace Functions
pub fn index(size: usize) -> VarRef {
    push_var(Var {
        op: Op::KernelOp(ir::Op::Index),
        deps: vec![],
        ty: VarType::U32,
        size,
        ..Default::default()
    })
}
pub fn sized_literal<T: AsVarType>(val: T, size: usize) -> VarRef {
    let ty = T::var_ty();
    let mut data = 0;
    unsafe { *(&mut data as *mut _ as *mut T) = val };
    push_var(Var {
        op: Op::KernelOp(ir::Op::Literal),
        ty,
        size,
        data: Data::Literal(data),
        ..Default::default()
    })
}
pub fn literal<T: AsVarType>(val: T) -> VarRef {
    sized_literal(val, 0)
}
fn max_size<'a>(refs: impl Iterator<Item = &'a VarRef>) -> usize {
    refs.map(|r| r.size()).reduce(|s0, s1| s0.max(s1)).unwrap()
}
pub fn composite(refs: &[&VarRef]) -> VarRef {
    let ty = VarType::Struct {
        tys: refs.iter().map(|r| r.ty()).collect(),
    };
    let size = max_size(refs.iter().map(|r| *r));
    let deps = refs.iter().map(|r| r.id()).collect::<Vec<_>>();
    push_var(Var {
        op: Op::KernelOp(ir::Op::Construct),
        deps,
        ty,
        size,
        ..Default::default()
    })
}
pub fn vec(refs: &[&VarRef]) -> VarRef {
    // TODO: validate
    //
    let ty = refs[0].ty();
    let size = max_size(refs.iter().map(|r| *r));

    let ty = VarType::Vec {
        ty: Box::new(ty),
        num: refs.len(),
    };
    let deps = refs.iter().map(|r| r.id()).collect::<Vec<_>>();
    push_var(Var {
        op: Op::KernelOp(ir::Op::Construct),
        deps,
        ty,
        size,
        ..Default::default()
    })
}
impl VarRef {
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

impl VarRef {
    pub fn add(&self, other: &VarRef) -> VarRef {
        assert_eq!(self._thread_id, std::thread::current().id());
        assert_eq!(other._thread_id, std::thread::current().id());
        let info = with_trace(|t| t.var_info(&[self.id(), other.id()]));
        push_var(Var {
            op: Op::KernelOp(ir::Op::Bop(ir::Bop::Add)),
            deps: vec![self.id(), other.id()],
            ty: info.ty,
            size: info.size,
            ..Default::default()
        })
    }
    pub fn gather(&self, idx: &Self, active: &Self) -> Self {
        let ty = self.ty();
        let size = idx.size();
        let src_ref = self.get_ref();
        push_var(Var {
            op: Op::KernelOp(ir::Op::Gather),
            deps: vec![src_ref.id(), idx.id(), active.id()],
            ty,
            size,
            ..Default::default()
        })
    }
    pub fn scatter(&self, dst: &Self, idx: &Self, active: &Self) -> Self {
        // It is important that dst is schedules
        dst.schedule();
        let info = with_trace(|t| t.var_info(&[self.id(), idx.id()]));
        let dst_ref = dst.get_mut();
        let res = push_var(Var {
            op: Op::KernelOp(ir::Op::Scatter),
            deps: vec![dst_ref.id(), self.id(), idx.id(), active.id()],
            ty: VarType::Void,
            size: info.size,
            ..Default::default()
        });
        res.schedule(); // Auto schedule
        res
    }
    pub fn get_ref(&self) -> Self {
        self._get_ref(false)
    }
    pub fn get_mut(&self) -> Self {
        self._get_ref(true)
    }
    fn _get_ref(&self, mutable: bool) -> Self {
        let ty = self.ty();
        push_var(Var {
            op: Op::Ref { mutable },
            deps: vec![self.id()],
            ty,
            size: 0,
            ..Default::default()
        })
    }
    pub fn ty(&self) -> VarType {
        with_trace(|t| t.var(self.id()).ty.clone())
    }
    pub fn size(&self) -> usize {
        with_trace(|t| t.var(self.id()).size)
    }
    pub fn to_vec<T: AsVarType + bytemuck::Pod>(&self) -> Vec<T> {
        assert_eq!(self._thread_id, std::thread::current().id());
        with_trace(|t| t.var(self.id()).data.buffer().unwrap().to_host().unwrap())
    }
    pub fn extract(&self, elem: usize) -> Self {
        let size = self.size();
        let ty = self.ty();
        let ty = match ty {
            VarType::Vec { ty, .. } => ty.as_ref().clone(),
            VarType::Struct { tys } => tys[elem].clone(),
            _ => todo!(),
        };
        push_var(Var {
            op: Op::KernelOp(ir::Op::Extract(elem)),
            deps: vec![self.id()],
            ty,
            size,
            ..Default::default()
        })
    }
    pub fn tex_lookup(&self, pos: &[&VarRef]) -> Self {
        assert!(pos.len() >= 1 && pos.len() <= 3);

        let pos = vec(pos);

        let size = pos.size();

        let (shape, channels) = self.shape_channels();
        assert!(channels <= 4);
        let ty = self.ty();
        let ty = VarType::Vec {
            ty: Box::new(ty),
            num: channels,
        };
        let src_ref = self.get_ref();

        let composite = push_var(Var {
            op: Op::KernelOp(ir::Op::TexLookup),
            deps: vec![src_ref.id(), pos.id()],
            ty,
            size,
            ..Default::default()
        });
        composite
    }
    pub fn shape_channels(&self) -> ([usize; 3], usize) {
        with_trace(|trace| match trace.var(self.id()).op.resulting_op() {
            Op::Texture { shape, channels } => (shape, channels),
            _ => todo!(),
        })
    }
}

// Device Ops
impl VarRef {
    pub fn texture(&self, shape: &[usize], channels: usize) -> Self {
        self.schedule();

        let size = self.size();
        assert_eq!(shape.iter().fold(1, |a, b| a * b) * channels, size);
        assert_eq!(self.ty(), VarType::F32);

        let shape = [
            *shape.get(0).unwrap_or(&0),
            *shape.get(1).unwrap_or(&0),
            *shape.get(2).unwrap_or(&0),
        ];
        log::trace!("Recording Texture with {shape:?}");

        push_var(Var {
            op: Op::DeviceOp(DeviceOp::Buffer2Texture { shape, channels }),
            deps: vec![self.id()],
            ty: VarType::F32,
            size: 0,
            ..Default::default()
        })
    }
}
