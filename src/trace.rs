use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::Range;
use std::thread::ThreadId;

use crate::data::Data;
use crate::extent::Extent;
use crate::op::{DeviceOp, Op, ReduceOp};
use crate::vartype::{AsVarType, Instance, Intersection, VarType};
use crate::{backend, ir};
use crate::{compiler, graph};
use slotmap::{DefaultKey, SlotMap};

#[derive(Debug, Default)]
pub struct Schedule {
    pub var_set: HashSet<VarId>,
    pub vars: Vec<VarRef>,
    pub groups: Vec<Range<usize>>,
    pub start: usize,
}
impl Schedule {
    pub fn new_group(&mut self) {
        // Clear `dirty` mark on variables
        for i in self.groups.last().unwrap_or(&(0..0)).clone() {
            self.vars[i].clear_dirty();
        }
        let start = self.start;
        let end = self.vars.len();
        if start != end {
            self.groups.push(start..end);
            self.start = end;
        }
    }
}

thread_local! {
    pub static TRACE: RefCell<Trace> = RefCell::new(Default::default());
    pub static SCHEDULE: RefCell<Schedule> = RefCell::new(Default::default());
}

#[derive(Default, Debug)]
pub struct Trace {
    entries: SlotMap<DefaultKey, Entry>,
}
impl Trace {
    pub fn var(&self, id: VarId) -> &Var {
        &self.entries[id.0].var
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.entries[id.0].var
    }
    pub fn entry_mut(&mut self, id: VarId) -> &mut Entry {
        &mut self.entries[id.0]
    }
    pub fn deps(&self, id: VarId) -> &[VarId] {
        &self.entries[id.0].deps
    }
    pub fn get_var(&mut self, id: VarId) -> Option<&Var> {
        self.entries.get(id.0).map(|entry| &entry.var)
    }
    pub fn push_var(&mut self, var: Var, deps: Vec<VarId>) -> VarId {
        for id in deps.iter() {
            self.inc_rc(*id);
        }
        let id = VarId(self.entries.insert(Entry { deps, var, rc: 1 }));
        id
    }
    pub fn inc_rc(&mut self, id: VarId) {
        self.entries[id.0].rc += 1;
    }
    pub fn dec_rc(&mut self, id: VarId) {
        let var = self.var_mut(id);
        let entry = &mut self.entries[id.0];
        entry.rc -= 1;
        if entry.rc == 0 {
            for dep in entry.deps.clone() {
                self.dec_rc(dep);
            }
            let var = self.var_mut(id);
            self.entries.remove(id.0);
        }
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
        assert_eq!(self.entries.len(), 0, "{self:#?}");
    }
}

#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(DefaultKey);

pub struct VarRef {
    id: VarId,
    _thread_id: ThreadId,
}
impl Debug for VarRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarRef").field("id", &self.id).finish()
    }
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
pub struct Entry {
    pub(crate) deps: Vec<VarId>,
    pub(crate) var: Var,
    pub(crate) rc: usize,
}

#[derive(Debug, Default)]
pub struct Var {
    pub op: Op,         // Operation used to construct the variable
    pub ty: VarType,    // Type of the variable
    pub extent: Extent, // Extent of the variable

    pub dirty: bool,

    pub data: Data,
}
pub fn with_trace<T, F: FnOnce(&mut Trace) -> T>(f: F) -> T {
    TRACE.with(|t| {
        let mut t = t.borrow_mut();
        f(&mut t)
    })
}
fn push_var<'a>(v: Var, deps: impl IntoIterator<Item = &'a VarRef>) -> VarRef {
    // Auto schedule_eval device ops
    let is_device_op = v.op.is_device_op();
    let deps = deps
        .into_iter()
        .map(|r| {
            if is_device_op {
                r.schedule();
            }
            r.id()
        })
        .collect::<Vec<_>>();

    // Schedule evaluation in some cases
    if is_device_op {
        schedule_eval();
    }
    if with_trace(|trace| deps.iter().any(|id| trace.var(*id).dirty)) {
        schedule_eval();
    }

    // Push actual variable
    let res = with_trace(|t| VarRef {
        id: t.push_var(v, deps),
        _thread_id: std::thread::current().id(),
    });
    // Auto schedule and schedule evaluation if device op
    if is_device_op {
        res.schedule();
        schedule_eval();
    }
    // TODO: maybe mark variables dirty that have been referenced with RefMut
    res
}

pub fn compile() -> graph::Graph {
    schedule_eval();
    SCHEDULE.with(|s| {
        let mut s = s.borrow_mut();
        let schedule = std::mem::take(&mut (*s));
        let graph = with_trace(|t| graph::compile(t, schedule));
        graph
    })
}
pub fn schedule_eval() {
    SCHEDULE.with(|s| {
        let mut s = s.borrow_mut();
        s.new_group();
    })
}

// Trace Functions
pub fn index(size: usize) -> VarRef {
    push_var(
        Var {
            op: Op::KernelOp(ir::Op::Index),
            ty: VarType::U32,
            extent: Extent::Size(size),
            ..Default::default()
        },
        [],
    )
}
pub fn literal<T: AsVarType>(val: T) -> VarRef {
    sized_literal(val, 0)
}
pub fn sized_literal<T: AsVarType>(val: T, size: usize) -> VarRef {
    let ty = T::var_ty();
    let mut data = 0;
    unsafe { *(&mut data as *mut _ as *mut T) = val };
    push_var(
        Var {
            op: Op::KernelOp(ir::Op::Literal),
            ty: ty.clone(),
            extent: Extent::Size(size),
            data: Data::Literal(data),
            ..Default::default()
        },
        [],
    )
}
pub fn array<T: AsVarType>(slice: &[T], device: &backend::Device) -> VarRef {
    let ty = T::var_ty();
    let size = slice.len();
    let slice: &[u8] =
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const _, slice.len() * ty.size()) };
    let data = device.create_buffer_from_slice(slice).unwrap();
    push_var(
        Var {
            op: Op::Buffer,
            extent: Extent::Size(size),
            ty: ty.clone(),
            data: Data::Buffer(data),
            ..Default::default()
        },
        [],
    )
}
fn resulting_extent<'a>(refs: impl IntoIterator<Item = &'a VarRef>) -> Extent {
    refs.into_iter()
        .map(|r| r.extent())
        .fold(Default::default(), |a, b| a.resulting_extent(&b))
}
pub fn composite(refs: &[&VarRef]) -> VarRef {
    let tys = refs.iter().map(|r| r.ty()).collect();
    let ty = VarType::Struct { tys };

    log::trace!("Constructing composite struct {ty:?}.");

    let extent = resulting_extent(refs.iter().map(|r| *r));
    push_var(
        Var {
            op: Op::KernelOp(ir::Op::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.into_iter().cloned(),
    )
}
pub fn vec(refs: &[&VarRef]) -> VarRef {
    // TODO: validate
    //
    let ty = refs[0].ty();

    let extent = resulting_extent(refs.iter().map(|r| *r));

    let ty = VarType::Vec {
        ty: Box::new(ty),
        num: refs.len(),
    };
    push_var(
        Var {
            op: Op::KernelOp(ir::Op::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.iter().cloned(),
    )
}

#[derive(Debug, Clone)]
pub enum GeometryDesc {
    Triangles { triangles: VarRef, vertices: VarRef },
}
#[derive(Debug, Clone)]
pub struct AccelDesc {
    pub geometries: Vec<GeometryDesc>,
    pub instances: VarRef,
}

pub fn accel(desc: &AccelDesc) -> VarRef {
    assert_eq!(&desc.instances.ty(), Instance::var_ty());
    let mut deps = vec![];
    deps.push(&desc.instances);
    let geometries = desc
        .geometries
        .iter()
        .map(|g| match g {
            GeometryDesc::Triangles {
                triangles,
                vertices,
            } => {
                // WARN: order in which triangles/vertices are pushed must match how they are
                // used when building Accel
                // triangles.schedule();
                // vertices.schedule();

                deps.push(triangles);
                deps.push(vertices);
                backend::GeometryDesc::Triangles {
                    n_triangles: triangles.size(),
                    n_vertices: vertices.size(),
                }
            }
        })
        .collect::<Vec<_>>();
    let create_desc = backend::AccelDesc {
        geometries,
        instances: desc.instances.size(),
    };

    push_var(
        Var {
            op: Op::DeviceOp(DeviceOp::BuildAccel),
            ty: VarType::Void,
            extent: Extent::Accel(create_desc.clone()),
            // accel_desc: Some(create_desc),
            ..Default::default()
        },
        deps,
    )
}

impl VarRef {
    pub fn mark_dirty(&self) {
        with_trace(|trace| {
            trace.var_mut(self.id()).dirty = true;
        })
    }
    fn clear_dirty(&self) {
        with_trace(|trace| {
            trace.var_mut(self.id()).dirty = false;
        })
    }
    pub fn dirty(&self) -> bool {
        with_trace(|trace| trace.var(self.id()).dirty)
    }
    pub fn same_trace(&self, other: &VarRef) -> bool {
        self._thread_id == other._thread_id
    }
    pub fn schedule(&self) {
        if self.evaluated() {
            return;
        }
        SCHEDULE.with(|s| {
            let mut s = s.borrow_mut();
            if !s.var_set.contains(&self.id()) {
                s.vars.push(self.clone());
                s.var_set.insert(self.id());
            }
        })
    }
    pub fn evaluated(&self) -> bool {
        with_trace(|trace| trace.var(self.id()).op.evaluated())
    }
    pub fn rc(&self) -> usize {
        with_trace(|trace| trace.entries[self.id().0].rc)
    }
}
macro_rules! bop {
    ($op:ident $(-> $result_type:expr)?) => {
        paste::paste! {
            pub fn $op(&self, other: &VarRef) -> VarRef {
                assert_eq!(self._thread_id, std::thread::current().id());
                assert_eq!(other._thread_id, std::thread::current().id());

                let extent = resulting_extent([self, other]);

                let ty = self.ty();
                assert_eq!(other.ty(), ty);

                $(let ty = $result_type;)?
                push_var(
                    Var {
                        op: Op::KernelOp(ir::Op::Bop(ir::Bop::[<$op:camel>])),
                        extent,
                        ty,
                        ..Default::default()
                    },
                    [self, other],
                )
            }
        }
    };
}
macro_rules! uop {
    ($op:ident $(-> $result_type:expr)?) => {
        paste::paste! {
            pub fn $op(&self) -> VarRef {
                assert_eq!(self._thread_id, std::thread::current().id());

                let extent = resulting_extent([self]);

                let ty = self.ty();

                $(let ty = $result_type;)?
                push_var(
                    Var {
                        op: Op::KernelOp(ir::Op::Uop(ir::Uop::[<$op:camel>])),
                        extent,
                        ty,
                        ..Default::default()
                    },
                    [self],
                )
            }
        }
    };
}
impl VarRef {
    // Binary operations returing the same type
    bop!(add);
    bop!(sub);
    bop!(mul);
    bop!(div);
    bop!(modulus);
    bop!(min);
    bop!(max);

    // Bitwise
    // TODO: more asserts for binary operations
    bop!(and);
    bop!(or);
    bop!(xor);

    // Shift
    // TODO: more asserts for shift operations
    bop!(shl);
    bop!(shr);

    // Comparisons
    bop!(eq -> VarType::Bool);
    bop!(neq -> VarType::Bool);
    bop!(lt -> VarType::Bool);
    bop!(le -> VarType::Bool);
    bop!(gt -> VarType::Bool);
    bop!(ge -> VarType::Bool);

    uop!(neg);
    uop!(sqrt);
    uop!(abs);
    uop!(sin);
    uop!(cos);
    uop!(exp2);
    uop!(log2);

    pub fn cast(&self, ty: VarType) -> Self {
        assert_eq!(self._thread_id, std::thread::current().id());

        let extent = resulting_extent([self]);

        push_var(
            Var {
                op: Op::KernelOp(ir::Op::Uop(ir::Uop::Cast)),
                extent,
                ty,
                ..Default::default()
            },
            [self],
        )
    }

    pub fn bitcast(&self, ty: VarType) -> Self {
        assert_eq!(self._thread_id, std::thread::current().id());

        let extent = resulting_extent([self]);

        push_var(
            Var {
                op: Op::KernelOp(ir::Op::Uop(ir::Uop::BitCast)),
                extent,
                ty,
                ..Default::default()
            },
            [self],
        )
    }

    pub fn gather(&self, idx: &Self) -> Self {
        self.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = idx.extent();
        let src_ref = self.get_ref();
        push_var(
            Var {
                op: Op::KernelOp(ir::Op::Gather),
                ty,
                extent,
                ..Default::default()
            },
            [&src_ref, idx],
        )
    }
    pub fn gather_if(&self, idx: &Self, active: &Self) -> Self {
        self.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = idx.extent();
        let src_ref = self.get_ref();
        push_var(
            Var {
                op: Op::KernelOp(ir::Op::Gather),
                ty,
                extent,
                ..Default::default()
            },
            [&src_ref, idx, active],
        )
    }
    // WARN: keep in mind, that we should also update `scatter_if`, `scatter_reduce` and
    // `scatter_reduce_if`
    pub fn scatter(&self, dst: &Self, idx: &Self) -> Self {
        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(ir::Op::Scatter(None)),
                ty: VarType::Void,
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
        res
    }
    pub fn scatter_if(&self, dst: &Self, idx: &Self, active: &Self) -> Self {
        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, idx, active].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(ir::Op::Scatter(None)),
                ty: VarType::Void,
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx, active],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
        res
    }
    // pub fn scatter_reduce(&self, dst: &Self, idx: &Self, op: ReduceOp) -> Self {
    //     dst.schedule();
    //     schedule_eval();
    //     let extent = resulting_extent([self, idx].into_iter());
    //     let dst_ref = dst.get_mut();
    //     let res = push_var(
    //         Var {
    //             op: Op::KernelOp(ir::Op::Scatter(Some(op))),
    //             ty: VarType::Void,
    //             extent,
    //             ..Default::default()
    //         },
    //         [&dst_ref, self, idx],
    //     );
    //     dst.mark_dirty();
    //     res.schedule(); // Auto schedule
    //     res
    // }
    // pub fn scatter_reduce_if(&self, dst: &Self, idx: &Self, active: &Self, op: ReduceOp) -> Self {
    //     dst.schedule();
    //     schedule_eval();
    //     let extent = resulting_extent([self, idx, active].into_iter());
    //     let dst_ref = dst.get_mut();
    //     let res = push_var(
    //         Var {
    //             op: Op::KernelOp(ir::Op::Scatter(Some(op))),
    //             ty: VarType::Void,
    //             extent,
    //             ..Default::default()
    //         },
    //         [&dst_ref, self, idx, active],
    //     );
    //     dst.mark_dirty();
    //     res.schedule(); // Auto schedule
    //     res
    // }
    pub fn scatter_atomic(&self, dst: &Self, idx: &Self, op: ReduceOp) -> Self {
        dst.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = resulting_extent([self, idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(ir::Op::Scatter(Some(op))),
                ty,
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx],
        );
        dst.mark_dirty();
        // NOTE: do not schedule result of scatter_atomic
        res
    }
    pub fn scatter_atomic_if(&self, dst: &Self, idx: &Self, active: &Self, op: ReduceOp) -> Self {
        dst.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = resulting_extent([self, idx, active].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(ir::Op::Scatter(Some(op))),
                ty,
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx, active],
        );
        dst.mark_dirty();
        // NOTE: do not schedule result of scatter_atomic
        res
    }
    pub fn get_ref(&self) -> Self {
        // TODO: maybe do scheduling here?
        self._get_ref(false)
    }
    pub fn get_mut(&self) -> Self {
        // TODO: maybe do scheduling here?
        self._get_ref(true)
    }
    fn _get_ref(&self, mutable: bool) -> Self {
        let ty = self.ty();
        push_var(
            Var {
                op: Op::Ref { mutable },
                ty,
                extent: Extent::Size(0),
                ..Default::default()
            },
            [self],
        )
    }
    pub fn ty(&self) -> VarType {
        with_trace(|t| t.var(self.id()).ty.clone())
    }
    pub fn size(&self) -> usize {
        match self.extent() {
            Extent::Size(size) => size,
            _ => todo!(),
        }
    }
    pub fn extent(&self) -> Extent {
        with_trace(|t| t.var(self.id()).extent.clone())
    }
    pub fn to_vec<T: AsVarType>(&self) -> Vec<T> {
        assert_eq!(self._thread_id, std::thread::current().id());
        with_trace(|t| t.var(self.id()).data.buffer().unwrap().to_host().unwrap())
    }
    pub fn extract(&self, elem: usize) -> Self {
        let extent = self.extent();
        let ty = self.ty();
        let ty = match ty {
            VarType::Vec { ty, .. } => ty.as_ref().clone(),
            VarType::Struct { tys } => tys[elem].clone(),
            _ => todo!(),
        };
        push_var(
            Var {
                op: Op::KernelOp(ir::Op::Extract(elem)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn select(&self, true_val: &Self, false_val: &Self) -> Self {
        assert_eq!(self.ty(), VarType::Bool);
        assert_eq!(true_val.ty(), false_val.ty());

        let ty = true_val.ty();

        let extent = resulting_extent([self, true_val, false_val].into_iter());

        push_var(
            Var {
                op: Op::KernelOp(ir::Op::Select),
                ty,
                extent,
                ..Default::default()
            },
            [self, true_val, false_val],
        )
    }
    pub fn tex_lookup(&self, pos: &[&VarRef]) -> Self {
        assert!(pos.len() >= 1 && pos.len() <= 3);

        let pos = vec(pos);

        let extent = pos.extent();

        let (_, channels) = self.extent().shape_and_channles();
        assert!(channels <= 4);
        let ty = self.ty();
        let ty = VarType::Vec {
            ty: Box::new(ty),
            num: channels,
        };
        let src_ref = self.get_ref();

        let composite = push_var(
            Var {
                op: Op::KernelOp(ir::Op::TexLookup),
                ty,
                extent,
                ..Default::default()
            },
            [&src_ref, &pos],
        );
        composite
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

        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::Buffer2Texture),
                ty: VarType::F32,
                extent: Extent::Texture { shape, channels },
                ..Default::default()
            },
            [self],
        )
    }
    pub fn trace_ray(&self, o: &Self, d: &Self, tmin: &Self, tmax: &Self) -> Self {
        let extent = resulting_extent([o, d, tmin, tmax].into_iter());

        let ty = Intersection::var_ty();

        let accel_ref = self.get_ref();

        push_var(
            Var {
                op: Op::KernelOp(ir::Op::TraceRay),
                ty: ty.clone(),
                extent,
                ..Default::default()
            },
            [&accel_ref, o, d, tmin, tmax],
        )
    }
}
// Reduction Operations
impl VarRef {
    pub fn reduce_max(&self) -> Self {
        let extent = Extent::Size(1);
        let ty = self.ty();
        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::ReduceOp(ReduceOp::Max)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn reduce_min(&self) -> Self {
        let extent = Extent::Size(1);
        let ty = self.ty();
        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::ReduceOp(ReduceOp::Min)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn reduce_sum(&self) -> Self {
        let extent = Extent::Size(1);
        let ty = self.ty();
        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::ReduceOp(ReduceOp::Sum)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn reduce_prod(&self) -> Self {
        let extent = Extent::Size(1);
        let ty = self.ty();
        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::ReduceOp(ReduceOp::Prod)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn reduce_or(&self) -> Self {
        let extent = Extent::Size(1);
        let ty = self.ty();
        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::ReduceOp(ReduceOp::Or)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }

    pub fn reduce_and(&self) -> Self {
        let extent = Extent::Size(1);
        let ty = self.ty();
        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::ReduceOp(ReduceOp::And)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
}
