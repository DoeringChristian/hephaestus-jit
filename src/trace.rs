use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Mutex;
use std::thread::ThreadId;

use crate::extent::Extent;
use crate::graph;
use crate::op::{Bop, DeviceOp, KernelOp, Op, ReduceOp, Uop};
use crate::resource::Resource;
use crate::vartype::{self, AsVarType, Instance, Intersection, VarType};
use crate::{backend, utils};
use indexmap::IndexMap;
use once_cell::sync::Lazy;
use slotmap::{DefaultKey, SlotMap};

pub use crate::record::record;

/// This struct describes a set of variables, which are scheduled for evaluation as well as groups
/// of these variables, that can be evaluated at once.
/// Where mitsuba calles `dr.eval` for certain operations, which causes evaluation and potential
/// CPU side stalls, we just record these evaluations by calling [schedule_eval].
///
/// We also deduplicate scheduled variables using the [Self::var_set] hash set, allowing a variable
/// to only be scheduled once.
///
/// The equivalent struct in Dr.Jit would be `ThreadState`
///
#[derive(Debug)]
pub struct ThreadState {
    // Scheduled variables
    // TODO: maybe use IndexSet
    pub scheduled: IndexMap<VarId, VarRef>,
    // Groups of scheduled variables, that can be compiled into the same kernel
    pub groups: Vec<Range<usize>>,
    // Start of the next group
    pub start: usize,

    // Represents the current scope
    pub scope: usize,
}
impl ThreadState {
    pub fn new_group(&mut self) {
        // Clear `dirty` mark on variables
        for i in self.groups.last().unwrap_or(&(0..0)).clone() {
            self.scheduled[i].clear_dirty();
        }
        let start = self.start;
        let end = self.scheduled.len();
        if start != end {
            self.groups.push(start..end);
            self.start = end;
        }
    }
    pub fn scope(&self) -> ScopeId {
        ScopeId(self.scope)
    }
    pub fn new_scope(&mut self) -> ScopeId {
        self.scope = with_trace(|trace| trace.new_scope()).0;
        self.scope()
    }
}
impl Default for ThreadState {
    fn default() -> Self {
        Self {
            scheduled: Default::default(),
            groups: Default::default(),
            start: Default::default(),
            scope: with_trace(|trace| trace.new_scope()).0,
        }
    }
}

pub static TRACE: Lazy<Mutex<Trace>> = Lazy::new(|| Mutex::new(Trace::default()));

thread_local! {
    // pub static TRACE: RefCell<Trace> = RefCell::new(Default::default());
    pub static TS: RefCell<ThreadState> = RefCell::new(Default::default());
}

///
/// A Directed Acyclic Graph (DAG), representing the traced computations.
/// Variables are tracked using reference counters, similar to mitsuba.
///
/// The Dr.Jit equivalent would be the `State` struct
///
#[derive(Default, Debug)]
pub struct Trace {
    vars: SlotMap<DefaultKey, Var>,
    scope_ctr: usize,
}
impl Trace {
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }
    pub fn deps(&self, id: VarId) -> &[VarId] {
        &self.vars[id.0].deps
    }
    pub fn get_var(&mut self, id: VarId) -> Option<&Var> {
        self.vars.get(id.0)
    }
    pub fn get_var_mut(&mut self, id: VarId) -> Option<&mut Var> {
        self.vars.get_mut(id.0)
    }
    pub fn push_var(&mut self, mut var: Var) -> VarId {
        for id in var.deps.iter().chain(var.extent.get_dynamic().as_ref()) {
            self.inc_rc(*id);
        }
        var.rc = 1;
        let id = VarId(self.vars.insert(var));
        id
    }
    pub fn inc_rc(&mut self, id: VarId) {
        self.vars[id.0].rc += 1;
    }
    ///
    /// Decrement the reference count of an entry in the trace.
    /// If the [Entry::rc] reaches 0, delte the variable and decrement the [Entry::rc] values of
    /// it's dependencies.
    ///
    pub fn dec_rc(&mut self, id: VarId) {
        let var = &mut self.vars[id.0];
        var.rc -= 1;
        if var.rc == 0 {
            let deps = var.deps.clone().into_iter().chain(var.extent.get_dynamic());

            for dep in deps {
                self.dec_rc(dep);
            }

            self.vars.remove(id.0);
        }
    }
    pub fn ref_borrow(&mut self, id: VarId) -> VarRef {
        self.inc_rc(id);
        VarRef { id }
    }
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
    pub fn new_scope(&mut self) -> ScopeId {
        let scope = ScopeId(self.scope_ctr);
        self.scope_ctr += 1;
        scope
    }
}
pub fn is_empty() -> bool {
    with_trace(|trace| trace.is_empty())
}
impl Drop for Trace {
    fn drop(&mut self) {
        assert_eq!(self.vars.len(), 0, "{self:#?}");
    }
}

///
/// This struct represents the id to a variable in the trace.
/// It allows for internal references within the trace, without causing reference counters to
/// change when copying.
///
#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct VarId(DefaultKey);

/// TODO: maybe make non-null
#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScopeId(pub(crate) usize);

///
/// This is a wrapper over the [VarId] id struct.
/// In contrast to [VarId], [VarRef] causes reference counter increments/decrements on clone/drop
///
pub struct VarRef {
    id: VarId,
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
        Self { id: self.id }
    }
}
impl Drop for VarRef {
    fn drop(&mut self) {
        with_trace(|t| t.dec_rc(self.id()))
    }
}

///
/// The [Var] struct represents a variable in the trace.
/// Every variable is the result of some operation [Op] and might hold some [Resource].
///
#[derive(Debug)]
pub struct Var {
    pub op: Op, // Operation used to construct the variable
    // pub ty: VarType,    // Type of the variable
    pub ty: &'static VarType,
    pub extent: Extent, // Extent of the variable

    pub scope: ScopeId,

    pub dirty: bool,

    pub data: Resource,

    pub(crate) deps: Vec<VarId>,
    pub(crate) rc: usize,
}
impl Default for Var {
    fn default() -> Self {
        Self {
            op: Default::default(),
            ty: vartype::void(),
            extent: Default::default(),
            scope: Default::default(),
            dirty: Default::default(),
            data: Default::default(),
            deps: Default::default(),
            rc: Default::default(),
        }
    }
}
pub fn with_trace<T, F: FnOnce(&mut Trace) -> T>(f: F) -> T {
    let mut t = TRACE.lock().unwrap();
    f(&mut *t)
    // TRACE.with(|t| {
    //     let mut t = t.borrow_mut();
    //     f(&mut t)
    // })
}
///
/// This function is used to push a Variable ([Var]) to the trace.
/// It takes care of certain special cases, such as:
/// - scheduling device wide dependencies
/// - scheduling evaluation of the previous group if the variable represents a device wide operation
/// - scheduling evaluation of the previous group if the variable depends on 'dirty' variables
/// - scheduling evaluation of the current group if the variable represents a device wide operation
///
fn push_var<'a>(mut v: Var, deps: impl IntoIterator<Item = &'a VarRef>) -> VarRef {
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

    // Schedule the current group for evaluation if the variable is a device wide operation or it
    // depends on a variable marked dirty
    if is_device_op {
        schedule_eval();
    }
    if with_trace(|trace| deps.iter().any(|id| trace.var(*id).dirty)) {
        schedule_eval();
    }

    // Set scope as max between thread state and dependencies
    v.scope = [TS.with(|s| s.borrow().scope())]
        .into_iter()
        .chain(
            deps.iter()
                .map(|id| with_trace(|trace| trace.var(*id).scope)),
        )
        .max()
        .unwrap();

    // Set dependencies

    v.deps = deps;

    // Push actual variable
    let res = with_trace(|t| VarRef { id: t.push_var(v) });
    // Auto schedule and schedule evaluation if device op
    if is_device_op {
        res.schedule();
        schedule_eval();
    }
    // TODO: maybe mark variables dirty that have been referenced with RefMut
    res
}
pub fn new_scope() -> ScopeId{
    TS.with(|ts|{
        ts.borrow_mut().new_scope()
    })
}

///
/// Compiles the currently scheduled variables (see [Schedule]) into a [graph::Graph], which can be
/// launched on any device.
/// This captures the current environment (variables which are already evaluated).
///
pub fn compile() -> graph::Graph {
    schedule_eval();
    TS.with(|s| {
        let mut s = s.borrow_mut();
        let schedule = std::mem::take(&mut (*s));
        let graph = with_trace(|t| graph::compile(t, &schedule, &[]));
        graph
    })
}
///
/// Schedules the current group of scheduled variables for evaluation (see [Schedule]).
///
pub fn schedule_eval() {
    TS.with(|s| {
        let mut s = s.borrow_mut();
        s.new_group();
    })
}

// Trace Functions

///
/// Returns a variable that represents a global index within a kernel.
///
pub fn index(size: usize) -> VarRef {
    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Index),
            ty: u32::var_ty(),
            extent: Extent::Size(size),
            ..Default::default()
        },
        [],
    )
}
pub fn dynamic_index(capacity: usize, size: &VarRef) -> VarRef {
    // Have to schedule size variable
    // NOTE: the reason we do not need to track rcs for the size variable, is that it has to be
    // scheduled.
    // Therefore the schedule should own it and not drop it.
    size.schedule();
    schedule_eval();
    let id = size.id();

    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Index),
            ty: u32::var_ty(),
            extent: Extent::DynSize { capacity, size: id },
            ..Default::default()
        },
        [],
    )
}
///
/// Returns a variable representing a literal within a kernel.
/// In contrast to [sized_literal], it cannot be evaluated.
///
pub fn literal<T: AsVarType>(val: T) -> VarRef {
    let ty = T::var_ty();
    let mut data = 0;
    unsafe { *(&mut data as *mut _ as *mut T) = val };
    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Literal),
            ty,
            extent: Extent::None,
            data: Resource::Literal(data),
            ..Default::default()
        },
        [],
    )
}
///
/// Returns a variable representing a literal within a kernel.
/// This operation also has an inpact on the size of variables depending on it.
/// It might be used to initialize buffers for device operations.
///
pub fn sized_literal<T: AsVarType>(val: T, size: usize) -> VarRef {
    let ty = T::var_ty();
    let mut data = 0;
    unsafe { *(&mut data as *mut _ as *mut T) = val };
    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Literal),
            ty,
            extent: Extent::Size(size),
            data: Resource::Literal(data),
            ..Default::default()
        },
        [],
    )
}
///
/// Returns a variable, representing an array on the device.
/// It requires a reference to the device to which the data should be uploaded and is the only user
/// space function to do so.
///
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
            ty,
            data: Resource::Buffer(data),
            ..Default::default()
        },
        [],
    )
}
///
/// Returns the [Extent], resulting from an operation on some variables.
///
fn resulting_extent<'a>(refs: impl IntoIterator<Item = &'a VarRef>) -> Extent {
    refs.into_iter()
        .map(|r| r.extent())
        .fold(Default::default(), |a, b| a.resulting_extent(&b))
}
///
/// Creates a composite struct, over a number of variables.
///
pub fn composite(refs: &[&VarRef]) -> VarRef {
    let tys = refs.iter().map(|r| r.ty()).collect::<Vec<_>>();
    let ty = vartype::composite(&tys);

    log::trace!("Constructing composite struct {ty:?}.");

    let extent = resulting_extent(refs.iter().map(|r| *r));
    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.into_iter().cloned(),
    )
}
pub fn arr(refs: &[&VarRef]) -> VarRef {
    // TODO: validate
    let ty = refs[0].ty();

    let extent = resulting_extent(refs.iter().map(|r| *r));

    let ty = vartype::array(ty, refs.len());

    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.iter().cloned(),
    )
}
///
/// Returns a variable of type [VarType::Vec], from the elements given.
/// The input variables should all have the same type.
///
pub fn vec(refs: &[&VarRef]) -> VarRef {
    // TODO: validate
    //
    let ty = refs[0].ty();

    let extent = resulting_extent(refs.iter().map(|r| *r));

    let ty = vartype::vector(ty, refs.len());

    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.iter().cloned(),
    )
}

pub fn mat(columns: &[&VarRef]) -> VarRef {
    // TODO: valiate
    let ty = columns[0].ty();
    let cols = columns.len();
    let (ty, rows) = match ty {
        VarType::Vec { ty, num } => (*ty, *num),
        _ => todo!(),
    };

    let extent = resulting_extent(columns.iter().map(|r| *r));

    let ty = vartype::matrix(ty, cols, rows);

    push_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        columns.iter().cloned(),
    )
}

#[derive(Debug, Clone)]
pub enum GeometryDesc {
    Triangles { triangles: VarRef, vertices: VarRef },
}
#[derive(Debug, Clone)]
pub struct AccelDesc<'a> {
    pub geometries: &'a [GeometryDesc],
    pub instances: VarRef,
}

pub fn accel(desc: &AccelDesc) -> VarRef {
    assert_eq!(desc.instances.ty(), Instance::var_ty());
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

                let capacity = vertices.capacity();
                let n_vertices = match vertices.ty(){
                    VarType::F32 if capacity % 3 == 0=> capacity/3,
                    VarType::Vec { ty, num } if *ty == f32::var_ty() && *num == 3 => capacity,
                    VarType::Array { ty, num } if *ty == f32::var_ty() && *num == 3 => capacity,
                    _ => todo!()
                };
                    
                let capacity = triangles.capacity();
                    dbg!(triangles.ty());
                let n_triangles = match triangles.ty(){
                    VarType::U32 if capacity % 3 == 0=> capacity/3,
                    VarType::Vec { ty, num } if *ty == u32::var_ty() && *num == 3 => capacity,
                    VarType::Array { ty, num } if *ty == u32::var_ty() && *num == 3 => capacity,
                    _ => todo!()
                };
                    
                log::trace!("Adding triangle geometry with max {n_triangles} triangles and max {n_vertices} vertices");

                deps.push(triangles);
                deps.push(vertices);
                backend::GeometryDesc::Triangles {
                    n_triangles,
                    n_vertices
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
            ty: vartype::void(),
            extent: Extent::Accel(create_desc.clone()),
            ..Default::default()
        },
        deps,
    )
}

impl VarRef {
    pub fn borrow(id: VarId) -> Self {
        with_trace(|trace| {
            trace.inc_rc(id);
        });
        Self { id }
    }
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
    /// Schedule the variable for execution in the current group
    pub fn schedule(&self) {
        // We should not be able to schedule already evaluated variables as well as ones who's
        // extent is unsized
        if self.is_evaluated() || self.is_unsized() {
            return;
        }
        TS.with(|s| {
            let mut s = s.borrow_mut();
            s.scheduled.entry(self.id()).or_insert_with(|| self.clone());
        })
    }
    pub fn is_evaluated(&self) -> bool {
        with_trace(|trace| trace.var(self.id()).op.evaluated())
    }
    pub fn is_unsized(&self) -> bool {
        with_trace(|trace| trace.var(self.id()).extent.is_unsized())
    }
    pub fn rc(&self) -> usize {
        with_trace(|trace| trace.vars[self.id().0].rc)
    }
}
macro_rules! bop {
    ($op:ident $(-> $result_type:expr)?) => {
        paste::paste! {
            pub fn $op(&self, other: &VarRef) -> VarRef {

                let extent = resulting_extent([self, other]);

                let ty = self.ty();
                assert_eq!(other.ty(), ty);

                $(let ty = $result_type;)?
                push_var(
                    Var {
                        op: Op::KernelOp(KernelOp::Bop(Bop::[<$op:camel>])),
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

                let extent = resulting_extent([self]);

                let ty = self.ty();

                $(let ty = $result_type;)?
                push_var(
                    Var {
                        op: Op::KernelOp(KernelOp::Uop(Uop::[<$op:camel>])),
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
    bop!(eq -> bool::var_ty());
    bop!(neq -> bool::var_ty());
    bop!(lt -> bool::var_ty());
    bop!(le -> bool::var_ty());
    bop!(gt -> bool::var_ty());
    bop!(ge -> bool::var_ty());

    uop!(neg);
    uop!(sqrt);
    uop!(abs);
    uop!(sin);
    uop!(cos);
    uop!(exp2);
    uop!(log2);

    pub fn cast(&self, ty: &'static VarType) -> Self {
        let extent = resulting_extent([self]);

        push_var(
            Var {
                op: Op::KernelOp(KernelOp::Uop(Uop::Cast)),
                extent,
                ty,
                ..Default::default()
            },
            [self],
        )
    }
    // pub fn reinterpret(&self, ty: VarType) -> Self {
    //     // TODO: for indirect dispatch, we have to change the dynamic size
    //
    //     let bytesize = self.extent().capacity() * self.ty().size();
    //     assert!(bytesize % ty.size() == 0);
    //
    //     todo!()
    //     // assert!(self.size() * self.ty().size() % ty.size())
    // }

    pub fn bitcast(&self, ty: &'static VarType) -> Self {
        let extent = resulting_extent([self]);

        push_var(
            Var {
                op: Op::KernelOp(KernelOp::Uop(Uop::BitCast)),
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
                op: Op::KernelOp(KernelOp::Gather),
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
                op: Op::KernelOp(KernelOp::Gather),
                ty,
                extent,
                ..Default::default()
            },
            [&src_ref, idx, active],
        )
    }
    // WARN: keep in mind, that we should also update `scatter_if`, `scatter_reduce` and
    // `scatter_reduce_if`
    pub fn scatter(&self, dst: &Self, idx: &Self) {
        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(KernelOp::Scatter),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_if(&self, dst: &Self, idx: &Self, active: &Self) {
        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, idx, active].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(KernelOp::Scatter),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx, active],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_reduce(&self, dst: &Self, idx: &Self, op: ReduceOp) {
        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(KernelOp::ScatterReduce(op)),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_reduce_if(&self, dst: &Self, idx: &Self, active: &Self, op: ReduceOp) {
        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, idx, active].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(KernelOp::ScatterReduce(op)),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, idx, active],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_atomic(&self, dst: &Self, idx: &Self, op: ReduceOp) -> Self {
        dst.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = resulting_extent([self, idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(KernelOp::ScatterAtomic(op)),
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
                op: Op::KernelOp(KernelOp::ScatterAtomic(op)),
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
    pub fn atomic_inc(self: &Self, idx: &Self, active: &Self) -> Self {
        // Destination is self
        self.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = resulting_extent([active].into_iter());
        assert!(matches!(idx.extent(), Extent::None));

        let dst_ref = self.get_mut();
        let res = push_var(
            Var {
                op: Op::KernelOp(KernelOp::AtomicInc),
                ty,
                extent,
                ..Default::default()
            },
            [&dst_ref, idx, active],
        );
        self.mark_dirty();
        // NOTE: do not schedule result of scatter_atomic
        res
    }
    pub fn fma(&self, b: &Self, c: &Self) -> Self {
        let extent = resulting_extent([self, b, c]);
        let ty = self.ty();

        push_var(
            Var {
                op: Op::KernelOp(KernelOp::FMA),
                extent,
                ty,
                ..Default::default()
            },
            [self, b, c],
        )
    }
}
impl VarRef {
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
    pub fn ty(&self) -> &'static VarType {
        with_trace(|t| t.var(self.id()).ty)
    }
    pub fn size(&self) -> usize {
        match self.extent() {
            Extent::Size(size) => size,
            _ => todo!(),
        }
    }
    pub fn capacity(&self) -> usize {
        match self.extent() {
            Extent::Size(size) => size,
            Extent::DynSize { capacity, .. } => capacity,
            _ => todo!(),
        }
    }
    pub fn extent(&self) -> Extent {
        with_trace(|t| t.var(self.id()).extent.clone())
    }
    pub fn scope(&self) -> ScopeId {
        with_trace(|t| t.var(self.id()).scope)
    }
    pub fn item<T: AsVarType>(&self) -> T {
        assert_eq!(self.size(), 1);
        self.to_vec(0..1)[0]
    }
    pub fn to_vec<T: AsVarType>(&self, range: impl std::ops::RangeBounds<usize>) -> Vec<T> {
        let size = match self.extent() {
            Extent::Size(size) => size,
            Extent::DynSize { size: size_dep, .. } => {
                VarRef::borrow(size_dep).to_vec::<i32>(0..1)[0] as usize
            }
            _ => todo!(),
        };
        // Limit range bounds to range of buffer
        let range = utils::usize::limit_range_bounds(range, 0..size);
        if range.len() == 0 {
            return vec![];
        }
        assert!(range.start <= range.end);

        let ty = self.ty();
        let start_bytes = range.start * ty.size();
        let end_bytes = range.end * ty.size();

        let dst_ty_size = T::var_ty().size();

        assert!(start_bytes % dst_ty_size == 0);
        assert!(end_bytes % dst_ty_size == 0);
        let start = start_bytes / dst_ty_size;
        let end = end_bytes / dst_ty_size;

        with_trace(|t| {
            t.var(self.id())
                .data
                .buffer()
                .unwrap()
                .to_host(start..end)
                .unwrap()
        })
    }
    pub fn extract(&self, elem: usize) -> Self {
        let extent = self.extent();
        let ty = self.ty();
        let ty = match ty {
            VarType::Vec { ty, .. } => ty,
            VarType::Struct { tys } => tys[elem],
            VarType::Array { ty, .. } => ty,
            _ => todo!(),
        };
        push_var(
            Var {
                op: Op::KernelOp(KernelOp::Extract(elem)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn extract_dyn(&self, elem: &VarRef) -> Self {
        let extent = self.extent();
        let ty = self.ty();
        let ty = match ty {
            VarType::Array { ty, .. } => ty,
            _ => todo!(),
        };

        push_var(
            Var {
                op: Op::KernelOp(KernelOp::DynExtract),
                ty,
                extent,
                ..Default::default()
            },
            [self, elem],
        )
    }
    pub fn extract_all(&self) -> Vec<Self> {
        let n_elements = self.ty().num_elements().unwrap();
        (0..n_elements).map(|i| self.extract(i)).collect()
    }
    pub fn select(&self, true_val: &Self, false_val: &Self) -> Self {
        assert_eq!(self.ty(), bool::var_ty());
        assert_eq!(true_val.ty(), false_val.ty());

        let ty = true_val.ty();

        let extent = resulting_extent([self, true_val, false_val].into_iter());

        push_var(
            Var {
                op: Op::KernelOp(KernelOp::Select),
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
        let ty = vartype::vector(ty, channels);
        let src_ref = self.get_ref();

        let composite = push_var(
            Var {
                op: Op::KernelOp(KernelOp::TexLookup),
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
        assert_eq!(self.ty(), f32::var_ty());

        let shape = [
            *shape.get(0).unwrap_or(&0),
            *shape.get(1).unwrap_or(&0),
            *shape.get(2).unwrap_or(&0),
        ];
        log::trace!("Recording Texture with {shape:?}");

        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::Buffer2Texture),
                ty: f32::var_ty(),
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
                op: Op::KernelOp(KernelOp::TraceRay),
                ty: ty,
                extent,
                ..Default::default()
            },
            [&accel_ref, o, d, tmin, tmax],
        )
    }
}
impl VarRef {
    pub fn compress_dyn(&self) -> Self {
        // TODO: this is somewhat inefficient, a better way would be to get a view into the buffer
        // with the correct extent.
        let capacity = self.capacity();
        let (count, indices) = self.compress();
        let dyn_index = dynamic_index(capacity, &count);
        let indices = indices.gather(&dyn_index);
        indices
    }
    /// Get's the argument to true values of a boolean array
    /// Returns a tuple (count: u32, indices: [u32])
    /// TODO: add this to the SSA graph instead of scheduling it
    pub fn compress(&self) -> (Self, Self) {
        assert_eq!(self.ty(), bool::var_ty());
        // let size = self.size();
        let extent = self.extent();
        // TODO: find a way to generate uninitialized arrays in a deffered manner
        let count = sized_literal(0u32, 1);
        let index = sized_literal(0u32, extent.capacity());

        count.schedule();
        index.schedule();
        self.schedule();
        schedule_eval();

        let res = push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::Compress),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&index, &count, self],
        );
        // NOTE: auto sceduled by `push_var`

        (count, index)
    }
}
impl VarRef {
    pub fn prefix_sum(&self, inclusive: bool) -> Self {
        let size = self.size();
        let ty = self.ty();

        let res = push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::PrefixSum { inclusive }),
                ty,
                extent: Extent::Size(size),
                ..Default::default()
            },
            [self],
        );
        res
    }
}
// Reduction Operations
impl VarRef {
    pub fn reduce(&self, op: ReduceOp) -> Self {
        let extent = Extent::Size(1);
        let ty = self.ty();
        push_var(
            Var {
                op: Op::DeviceOp(DeviceOp::ReduceOp(op)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn reduce_max(&self) -> Self {
        self.reduce(ReduceOp::Max)
    }
    pub fn reduce_min(&self) -> Self {
        self.reduce(ReduceOp::Min)
    }
    pub fn reduce_sum(&self) -> Self {
        self.reduce(ReduceOp::Sum)
    }
    pub fn reduce_prod(&self) -> Self {
        self.reduce(ReduceOp::Prod)
    }
    pub fn reduce_or(&self) -> Self {
        self.reduce(ReduceOp::Or)
    }
    pub fn reduce_xor(&self) -> Self {
        self.reduce(ReduceOp::Xor)
    }
    pub fn reduce_and(&self) -> Self {
        self.reduce(ReduceOp::And)
    }
}
