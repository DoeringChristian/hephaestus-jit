use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use crate::extent::Extent;
use crate::op::{Bop, DeviceOp, KernelOp, Op, ReduceOp, Uop};
use crate::resource::{Resource, ResourceDesc};
use crate::vartype::{self, AsVarType, Instance, Intersection, VarType};
use crate::{backend, utils};
use crate::{graph, resource};
use half::f16;
use indexmap::IndexMap;
use once_cell::sync::Lazy;
use slotmap::{DefaultKey, SlotMap};

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
pub(crate) struct ThreadState {
    // Scheduled variables
    // TODO: maybe use IndexSet
    pub scheduled: IndexMap<VarId, VarRef>,
    // Groups of scheduled variables, that can be compiled into the same kernel
    pub groups: Vec<Range<usize>>,
    // Start of the next group
    pub start: usize,

    // Keeps a stack of side effect variables for recording loops.
    // These will be made dependencies of the loop.
    pub recorded_se_start: Vec<usize>,
    pub recorded_se: Vec<VarRef>,
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
}
impl Default for ThreadState {
    fn default() -> Self {
        Self {
            scheduled: Default::default(),
            groups: Default::default(),
            start: Default::default(),
            recorded_se_start: Default::default(),
            recorded_se: Default::default(),
        }
    }
}

pub(crate) static TRACE: Lazy<Mutex<Trace>> = Lazy::new(|| Mutex::new(Trace::default()));

thread_local! {
    // pub static TRACE: RefCell<Trace> = RefCell::new(Default::default());
    pub(crate) static TS: RefCell<ThreadState> = RefCell::new(Default::default());
}

///
/// A Directed Acyclic Graph (DAG), representing the traced computations.
/// Variables are tracked using reference counters, similar to mitsuba.
///
/// The Dr.Jit equivalent would be the `State` struct
///
#[derive(Default, Debug)]
pub(crate) struct Trace {
    vars: SlotMap<DefaultKey, Var>,
}
impl Trace {
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    // pub fn var_mut(&mut self, id: VarId) -> &mut Var {
    //     &mut self.vars[id.0]
    // }
    pub fn mark_dirty(&mut self, id: VarId) {
        self.vars[id.0].dirty = true;
    }
    pub fn clear_dirty(&mut self, id: VarId) {
        self.vars[id.0].dirty = false;
    }
    pub fn deps(&self, id: VarId) -> &[VarId] {
        &self.vars[id.0].deps
    }
    pub fn get_var(&mut self, id: VarId) -> Option<&Var> {
        self.vars.get(id.0)
    }
    // TODO: remove this
    pub fn get_var_mut(&mut self, id: VarId) -> Option<&mut Var> {
        self.vars.get_mut(id.0)
    }
    pub fn new_var_id(&mut self, mut var: Var) -> VarId {
        for id in var.deps.iter().chain(var.extent.get_dynamic().as_ref()) {
            self.inc_rc(*id);
        }
        var.rc = 1;
        let id = VarId(self.vars.insert(var));
        id
    }
    pub fn new_var(&mut self, mut var: Var) -> VarRef {
        VarRef { id: self.new_var_id(var) }
    }
    pub fn inc_rc(&mut self, id: VarId) {
        self.vars[id.0].rc += 1;
    }
    ///
    /// Put variable into it's evaluated state, removing dependencies and chaning its op type to
    /// the evaluated type.
    ///
    pub fn advance(&mut self, id: VarId) {
        let var = &mut self.vars[id.0];

        var.op = var.op.resulting_op();

        // Clear dependencies:
        let deps = std::mem::take(&mut var.deps);

        for dep in deps {
            self.dec_rc(dep);
        }
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
    pub fn reindex(&mut self, id: VarId, new_idx: VarId) -> Option<VarId>{
        
        let mut deps = {
            let var = self.var(id);

            if !var.data.is_none() {
                return None;
            }

            if matches!(var.op, Op::KernelOp(KernelOp::BufferRef | KernelOp::AccelRef | KernelOp::TextureRef{..})){
                return Some(id);
            }

            var.deps.clone()
        };
        
        let mut failed = -1;
        for (i, dep) in deps.iter_mut().enumerate(){
            let reindexed = self.reindex(*dep, new_idx);
            if let Some(reindexed) = reindexed{
                *dep = reindexed;
            }else{
                failed = i as i32;
            }
        }
        if failed >= 0{
            for i in 0..=failed{
                self.dec_rc(deps[i as usize]);
            }
            return None;
        }
        

        
        let var = self.var(id);
        let new_idx_var = self.var(new_idx);

        let res = if var.op == Op::KernelOp(KernelOp::Index){
            self.inc_rc(new_idx);
            Some(new_idx)
        }else{
            Some(
                self.new_var_id(Var{
                    op: var.op,
                    ty: var.ty,
                    extent: new_idx_var.extent.clone(),
                    deps,
                    ..Default::default()
                })
            )
        };
        res
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
// WARN: Recording relies on this hash function
impl Hash for VarRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        with_trace(|trace| {
            let var = trace.var(self.id);
            var.op.hash(state);
            var.ty.hash(state);
            var.extent.hash(state);
            if matches!(var.op, Op::KernelOp(KernelOp::Literal)) {
                if let Resource::Literal(lit) = var.data {
                    lit.hash(state);
                }
            }
            // TODO: maybe add other values to hash
        });
    }
}
impl From<&VarRef> for VarRef {
    fn from(value: &VarRef) -> Self {
        value.clone()
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

    // pub scope: ScopeId,
    pub dirty: bool,

    pub data: Resource,

    pub(crate) deps: Vec<VarId>,
    pub(crate) rc: usize,
}
impl Var {
    pub fn resource_desc(&self) -> Option<resource::ResourceDesc> {
        match &self.extent {
            Extent::Size(size) => Some(resource::ResourceDesc::BufferDesc(resource::BufferDesc {
                size: *size,
                ty: self.ty,
            })),
            Extent::DynSize { capacity, .. } => {
                Some(resource::ResourceDesc::BufferDesc(resource::BufferDesc {
                    size: *capacity,
                    ty: self.ty,
                }))
            }
            Extent::Texture { shape, channels } => {
                Some(resource::ResourceDesc::TextureDesc(resource::TextureDesc {
                    shape: *shape,
                    channels: *channels,
                    format: self.ty,
                }))
            }
            Extent::Accel(desc) => Some(resource::ResourceDesc::AccelDesc(desc.clone())),
            _ => None,
        }
    }
}
impl Default for Var {
    fn default() -> Self {
        Self {
            op: Default::default(),
            ty: vartype::void(),
            extent: Default::default(),
            // scope: Default::default(),
            dirty: Default::default(),
            data: Default::default(),
            deps: Default::default(),
            rc: Default::default(),
        }
    }
}
pub(crate) fn with_trace<T, F: FnOnce(&mut Trace) -> T>(f: F) -> T {
    let mut t = TRACE.lock().unwrap();
    f(&mut *t)
    // TRACE.with(|t| {
    //     let mut t = t.borrow_mut();
    //     f(&mut t)
    // })
}
///
/// This function is used to create a variable ([Var]) on the trace.
/// It takes care of certain special cases, such as:
/// - scheduling device wide dependencies
/// - scheduling evaluation of the previous group if the variable represents a device wide operation
/// - scheduling evaluation of the previous group if the variable depends on 'dirty' variables
/// - scheduling evaluation of the current group if the variable represents a device wide operation
///
pub(crate) fn new_var<'a>(mut v: Var, deps: impl IntoIterator<Item = &'a VarRef>) -> VarRef {
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

    // Set dependencies

    v.deps = deps;

    // Push actual variable
    let res = with_trace(|t| t.new_var(v));
    // Auto schedule and schedule evaluation if device op
    if is_device_op {
        res.schedule();
        schedule_eval();
    }
    // TODO: maybe mark variables dirty that have been referenced with RefMut
    res
}

// Loop stuff
///
/// Constructs the start of a recorded loop.
/// The state of the loop is handled through a construct.
/// We could introduce a 'namespace' cosntruct type, that would allow us to not actually use spir-v
/// variables.
///
pub fn loop_start(state_vars: &[VarRef]) -> (VarRef, impl Iterator<Item = VarRef>) {
    let state = composite(state_vars);

    let ty = state.ty();
    let extent = state.extent();

    // Start a new group of recorded side effects, to be used by loop_end
    TS.with(|ts| {
        let mut ts = ts.borrow_mut();
        let start = ts.recorded_se.len();
        ts.recorded_se_start.push(start);
    });

    let loop_start = new_var(
        Var {
            op: Op::KernelOp(KernelOp::LoopStart),
            ty,
            extent,
            ..Default::default()
        },
        [&state],
    );

    let state = loop_start.extract_all();

    (loop_start, state)
}
pub fn loop_end(loop_start: &VarRef, state_vars: &[VarRef]) -> impl Iterator<Item = VarRef> {
    let state = composite(state_vars);

    let ty = state.ty();
    let extent = state.extent();

    // Get side effects of this loop
    let side_effects = TS.with(|ts| {
        let mut ts = ts.borrow_mut();
        let start = ts.recorded_se_start.pop().unwrap();
        dbg!(start);
        ts.recorded_se.drain(start..).collect::<Vec<_>>()
    });

    let deps = [loop_start, &state].into_iter().chain(side_effects.iter());

    let loop_end = new_var(
        Var {
            op: Op::KernelOp(KernelOp::LoopEnd),
            ty,
            extent,
            ..Default::default()
        },
        deps,
    );

    let state = loop_end.extract_all();

    state
}
pub fn if_start(state_vars: &[VarRef]) -> (VarRef, impl Iterator<Item = VarRef>) {
    let state = composite(state_vars);

    let ty = state.ty();
    let extent = state.extent();

    TS.with(|ts| {
        let mut ts = ts.borrow_mut();
        let start = ts.recorded_se.len();
        ts.recorded_se_start.push(start);
    });

    let if_start = new_var(
        Var {
            op: Op::KernelOp(KernelOp::IfStart),
            ty,
            extent,
            ..Default::default()
        },
        [&state],
    );

    let state = if_start.extract_all();

    (if_start, state)
}

pub fn if_end(if_start: &VarRef, state_vars: &[VarRef]) -> impl Iterator<Item = VarRef> {
    let state = composite(state_vars);

    let ty = state.ty();
    let extent = state.extent();

    // Get side effects of this loop
    let side_effects = TS.with(|ts| {
        let mut ts = ts.borrow_mut();
        let start = ts.recorded_se_start.pop().unwrap();
        dbg!(start);
        ts.recorded_se.drain(start..).collect::<Vec<_>>()
    });

    let deps = [if_start, &state].into_iter().chain(side_effects.iter());

    let if_end = new_var(
        Var {
            op: Op::KernelOp(KernelOp::LoopEnd),
            ty,
            extent,
            ..Default::default()
        },
        deps,
    );

    let state = if_end.extract_all();

    state
}

///
/// Compiles the current thread state (see [ThreadState]) into a [graph::Graph], which can be
/// launched on any device.
/// This captures the current environment (variables which are already evaluated).
///
pub fn compile() -> graph::Result<graph::Graph> {
    schedule_eval();
    TS.with(|s| {
        let mut ts = s.borrow_mut();
        let ts = std::mem::take(&mut (*ts));
        let graph = graph::compile(&ts, &[], &[]);
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
pub fn index() -> VarRef {
    new_var(
        Var {
            op: Op::KernelOp(KernelOp::Index),
            ty: u32::var_ty(),
            extent: Default::default(),
            ..Default::default()
        },
        [],
    )
}
///
/// Returns a variable that represents a global index within a kernel, while enforcing a minimum
/// size.
///
pub fn sized_index(size: usize) -> VarRef {
    new_var(
        Var {
            op: Op::KernelOp(KernelOp::Index),
            ty: u32::var_ty(),
            extent: Extent::Size(size),
            ..Default::default()
        },
        [],
    )
}
pub fn dynamic_index(capacity: usize, size: impl Into<VarRef>) -> VarRef {
    let size = size.into();
    // Have to schedule size variable
    // NOTE: the reason we do not need to track rcs for the size variable, is that it has to be
    // scheduled.
    // Therefore the schedule should own it and not drop it.
    size.schedule();
    schedule_eval();
    let id = size.id();

    new_var(
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
    new_var(
        Var {
            op: Op::KernelOp(KernelOp::Literal),
            ty,
            extent: Default::default(),
            data: Resource::Literal(data),
            ..Default::default()
        },
        [],
    )
}
impl<T: AsVarType> From<T> for VarRef {
    fn from(value: T) -> Self {
        literal(value)
    }
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
    new_var(
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
    new_var(
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
pub fn composite(refs: &[VarRef]) -> VarRef {
    let tys = refs.iter().map(|r| r.ty()).collect::<Vec<_>>();
    let ty = vartype::composite(&tys);

    log::trace!("Constructing composite struct {ty:?}.");

    let extent = resulting_extent(refs.iter());
    new_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.into_iter(),
    )
}
pub fn arr(refs: &[VarRef]) -> VarRef {
    // TODO: validate
    let ty = refs[0].ty();

    let extent = resulting_extent(refs.iter());

    let ty = vartype::array(ty, refs.len());

    new_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.iter(),
    )
}
///
/// Returns a variable of type [VarType::Vec], from the elements given.
/// The input variables should all have the same type.
///
pub fn vec(refs: &[VarRef]) -> VarRef {
    // TODO: validate
    //
    let ty = refs[0].ty();

    let extent = resulting_extent(refs.iter());

    let ty = vartype::vector(ty, refs.len());

    new_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        refs.iter(),
    )
}

pub fn mat(columns: &[VarRef]) -> VarRef {
    // TODO: valiate
    let ty = columns[0].ty();
    let cols = columns.len();
    let (ty, rows) = match ty {
        VarType::Vec { ty, num } => (*ty, *num),
        _ => todo!(),
    };

    let extent = resulting_extent(columns.iter());

    let ty = vartype::matrix(ty, cols, rows);

    new_var(
        Var {
            op: Op::KernelOp(KernelOp::Construct),
            ty,
            extent,
            ..Default::default()
        },
        columns.iter(),
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
        .collect::<Arc<[_]>>();
    let create_desc = backend::AccelDesc {
        geometries,
        instances: desc.instances.size(),
    };

    new_var(
        Var {
            op: Op::DeviceOp(DeviceOp::BuildAccel),
            ty: vartype::void(),
            extent: Extent::Accel(create_desc.clone()),
            ..Default::default()
        },
        deps,
    )
}

#[allow(non_snake_case)]
pub fn matfma(
    mat_a: impl Into<VarRef>,
    mat_b: impl Into<VarRef>,
    mat_c: impl Into<VarRef>,
    M: usize,
    N: usize,
    K: usize,
) -> VarRef {
    let mat_a = mat_a.into();
    let mat_b = mat_b.into();
    let mat_c = mat_c.into();

    assert_eq!(mat_a.ty(), mat_b.ty());
    let c_type = mat_c.ty();

    let mat_c = new_var(
        Var {
            op: Op::DeviceOp(DeviceOp::MatMul {
                max_n: N,
                max_m: M,
                max_k: K,
            }),
            ty: c_type,
            extent: Extent::Size(N * M),
            ..Default::default()
        },
        [&mat_a, &mat_b, &mat_c],
    );

    mat_c
}

pub fn fused_mlp_inference(
    input: impl Into<VarRef>,
    weights: impl Into<VarRef>,
    width: usize,
    in_width: usize,
    out_width: usize,
    hidden_layers: usize,
    batch_size: usize,
) -> VarRef {
    let input = input.into();
    let weights = weights.into();

    assert_eq!(width, in_width);
    assert_eq!(width, out_width);
    assert_eq!(input.ty(), f16::var_ty());
    assert_eq!(weights.ty(), f16::var_ty());

    let ty = f16::var_ty();
    let size = input.size();

    new_var(
        Var {
            op: Op::DeviceOp(DeviceOp::FusedMlpInference {
                width,
                in_width,
                out_width,
                hidden_layers,
                max_batch_size: batch_size,
            }),
            ty,
            extent: Extent::Size(size),
            ..Default::default()
        },
        [&input, &weights],
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
            trace.mark_dirty(self.id());
        })
    }
    fn clear_dirty(&self) {
        with_trace(|trace| {
            trace.clear_dirty(self.id());
        })
    }
    pub fn dirty(&self) -> bool {
        with_trace(|trace| trace.var(self.id()).dirty)
    }
    pub fn schedule(&self) {
        // We should not be able to schedule already evaluated variables as well as ones who's
        // extent is unsized
        if self.is_evaluated() || self.is_unsized() {
            return;
        }
        TS.with(|s| {
            let mut ts = s.borrow_mut();
            if !ts.recorded_se_start.is_empty() && self.ty() == vartype::void() {
                ts.recorded_se.push(self.clone());
            } else {
                ts.scheduled
                    .entry(self.id())
                    .or_insert_with(|| self.clone());
            }
        })
    }
    pub fn is_evaluated(&self) -> bool {
        with_trace(|trace| trace.var(self.id()).op.evaluated())
    }
    pub fn is_unsized(&self) -> bool {
        with_trace(|trace| trace.var(self.id()).extent.is_unsized())
    }
    pub fn has_deps(&self) -> bool{
        !with_trace(|trace| trace.var(self.id()).deps.is_empty())
    }
    pub fn rc(&self) -> usize {
        with_trace(|trace| trace.vars[self.id().0].rc)
    }
    pub fn is_data(&self) -> bool{
        !with_trace(|trace| trace.var(self.id()).data.is_none())
    }
    pub fn is_ref(&self) -> bool{
        with_trace(|trace| matches!(trace.var(self.id()).op, Op::KernelOp(KernelOp::BufferRef | KernelOp::AccelRef | KernelOp::TextureRef{..})))
    }
    pub fn deps(&self) -> Vec<VarRef>{
        with_trace(|trace|{
            // TODO: remove clone by unsafe
            trace.var(self.id()).deps.clone().into_iter().map(|dep|{
                trace.ref_borrow(dep)
            }).collect()
        })
    }
    pub fn op(&self) -> Op{
        with_trace(|trace|{
            trace.var(self.id()).op
        })
    }
}
macro_rules! bop {
    ($op:ident $(-> $result_type:expr)?) => {
        paste::paste! {
            pub fn $op(&self, rhs: impl Into<VarRef>) -> VarRef {
                let rhs = rhs.into();

                let extent = resulting_extent([self, &rhs]);

                let ty = self.ty();
                assert_eq!(rhs.ty(), ty);

                $(let ty = $result_type;)?
                new_var(
                    Var {
                        op: Op::KernelOp(KernelOp::Bop(Bop::[<$op:camel>])),
                        extent,
                        ty,
                        ..Default::default()
                    },
                    [self, &rhs],
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
                new_var(
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
    // Arithmetic
    bop!(add);
    bop!(sub);
    bop!(mul);
    bop!(div);
    bop!(modulus);
    bop!(min);
    bop!(max);

    // Vector
    bop!(inner);

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

        new_var(
            Var {
                op: Op::KernelOp(KernelOp::Uop(Uop::Cast)),
                extent,
                ty,
                ..Default::default()
            },
            [self],
        )
    }

    pub fn bitcast(&self, ty: &'static VarType) -> Self {
        let extent = resulting_extent([self]);

        new_var(
            Var {
                op: Op::KernelOp(KernelOp::Uop(Uop::BitCast)),
                extent,
                ty,
                ..Default::default()
            },
            [self],
        )
    }

    pub fn reindex(&self, new_idx: &VarRef) -> Option<VarRef>{

        if self.is_data(){
            return None;
        }

        if self.is_ref(){
            return Some(self.clone());
        }

        let mut deps = self.deps();
        
        for dep in deps.iter_mut(){
            *dep = dep.reindex(new_idx)?;
        }
        
        // let var = self.var(id);
        // let new_idx_var = self.var(new_idx);
        let op = self.op();
        let ty = self.ty();
        let extent = new_idx.extent();
        let data = self.data();

        let res = if op == Op::KernelOp(KernelOp::Index){
            Some(new_idx.clone())
        }else{
            Some(
                new_var(Var{
                    op,
                    ty,
                    extent,
                    data,
                    ..Default::default()
                }, deps.iter())
            )
        };
        res
    }
    pub fn gather(&self, idx: impl Into<Self>) -> Self {
        self.gather_if(idx, true)
    }
    pub fn gather_if(&self, idx: impl Into<Self>, active: impl Into<Self>) -> Self {
        let idx = idx.into();
        let active = active.into();

        // Resize if the source variable is a literal
        if self.is_unsized(){
            return with_trace(|trace|{
                let var = trace.var(self.id());
                let extent = trace.var(idx.id()).extent.clone();
                let  var = Var{
                    op: var.op,
                    ty: var.ty,
                    extent,
                    data: var.data.clone(),
                    ..Default::default()
                };
                trace.new_var(var)
            });
        }

        // Reindex if possible
        if let Some(reindexed) = self.reindex(&idx){
            return reindexed;
        }
        
        self.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = idx.extent();
        let src_ref = self.get_ref();
        new_var(
            Var {
                op: Op::KernelOp(KernelOp::Gather),
                ty,
                extent,
                ..Default::default()
            },
            [&src_ref, &idx, &active],
        )
    }
    // WARN: keep in mind, that we should also update `scatter_if`, `scatter_reduce` and
    // `scatter_reduce_if`
    pub fn scatter(&self, dst: impl Into<Self>, idx: impl Into<Self>) {
        let dst = dst.into();
        let idx = idx.into();

        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, &idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = new_var(
            Var {
                op: Op::KernelOp(KernelOp::Scatter),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, &idx],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_if(&self, dst: impl Into<Self>, idx: impl Into<Self>, active: impl Into<Self>) {
        let dst = dst.into();
        let idx = idx.into();
        let active = active.into();

        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, &idx, &active].into_iter());
        let dst_ref = dst.get_mut();
        let res = new_var(
            Var {
                op: Op::KernelOp(KernelOp::Scatter),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, &idx, &active],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_reduce(&self, dst: impl Into<Self>, idx: impl Into<Self>, op: ReduceOp) {
        let dst = dst.into();
        let idx = idx.into();

        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, &idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = new_var(
            Var {
                op: Op::KernelOp(KernelOp::ScatterReduce(op)),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, &idx],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_reduce_if(
        &self,
        dst: impl Into<Self>,
        idx: impl Into<Self>,
        active: impl Into<Self>,
        op: ReduceOp,
    ) {
        let dst = dst.into();
        let idx = idx.into();
        let active = active.into();

        dst.schedule();
        schedule_eval();
        let extent = resulting_extent([self, &idx, &active].into_iter());
        let dst_ref = dst.get_mut();
        let res = new_var(
            Var {
                op: Op::KernelOp(KernelOp::ScatterReduce(op)),
                ty: vartype::void(),
                extent,
                ..Default::default()
            },
            [&dst_ref, self, &idx, &active],
        );
        dst.mark_dirty();
        res.schedule(); // Auto schedule
                        // res
    }
    pub fn scatter_atomic(&self, dst: impl Into<Self>, idx: impl Into<Self>, op: ReduceOp) -> Self {
        let dst = dst.into();
        let idx = idx.into();

        dst.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = resulting_extent([self, &idx].into_iter());
        let dst_ref = dst.get_mut();
        let res = new_var(
            Var {
                op: Op::KernelOp(KernelOp::ScatterAtomic(op)),
                ty,
                extent,
                ..Default::default()
            },
            [&dst_ref, self, &idx],
        );
        dst.mark_dirty();
        // NOTE: do not schedule result of scatter_atomic
        res
    }
    pub fn scatter_atomic_if(
        &self,
        dst: impl Into<Self>,
        idx: impl Into<Self>,
        active: impl Into<Self>,
        op: ReduceOp,
    ) -> Self {
        let dst = dst.into();
        let idx = idx.into();
        let active = active.into();

        dst.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = resulting_extent([self, &idx, &active].into_iter());
        let dst_ref = dst.get_mut();
        let res = new_var(
            Var {
                op: Op::KernelOp(KernelOp::ScatterAtomic(op)),
                ty,
                extent,
                ..Default::default()
            },
            [&dst_ref, self, &idx, &active],
        );
        dst.mark_dirty();
        // NOTE: do not schedule result of scatter_atomic
        res
    }
    pub fn atomic_inc(&self, idx: impl Into<Self>, active: impl Into<Self>) -> Self {
        let idx = idx.into();
        let active = active.into();

        // Destination is self
        self.schedule();
        schedule_eval();
        let ty = self.ty();
        let extent = resulting_extent([&active].into_iter());
        assert!(matches!(idx.extent(), Extent::Size(size) if size <= 1));

        let dst_ref = self.get_mut();
        let res = new_var(
            Var {
                op: Op::KernelOp(KernelOp::AtomicInc),
                ty,
                extent,
                ..Default::default()
            },
            [&dst_ref, &idx, &active],
        );
        self.mark_dirty();
        // NOTE: do not schedule result of scatter_atomic
        res
    }
    pub fn fma(&self, b: impl Into<Self>, c: impl Into<Self>) -> Self {
        let b = b.into();
        let c = c.into();

        let extent = resulting_extent([self, &b, &c]);
        let ty = self.ty();

        new_var(
            Var {
                op: Op::KernelOp(KernelOp::FMA),
                extent,
                ty,
                ..Default::default()
            },
            [self, &b, &c],
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
        new_var(
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
            // Extent::None => 0,
            Extent::Size(size) => size,
            Extent::DynSize { capacity, .. } => capacity,
            _ => todo!(),
        }
    }
    pub fn extent(&self) -> Extent {
        with_trace(|t| t.var(self.id()).extent.clone())
    }
    pub fn data(&self) -> Resource{
        with_trace(|t| t.var(self.id()).data.clone())
    }
    // pub fn scope(&self) -> ScopeId {
    //     with_trace(|t| t.var(self.id()).scope)
    // }
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
        new_var(
            Var {
                op: Op::KernelOp(KernelOp::Extract(elem as u32)),
                ty,
                extent,
                ..Default::default()
            },
            [self],
        )
    }
    pub fn extract_dyn(&self, elem: impl Into<Self>) -> Self {
        let elem = elem.into();

        let extent = self.extent();
        let ty = self.ty();
        let ty = match ty {
            VarType::Array { ty, .. } => ty,
            _ => todo!(),
        };

        new_var(
            Var {
                op: Op::KernelOp(KernelOp::DynExtract),
                ty,
                extent,
                ..Default::default()
            },
            [self, &elem],
        )
    }
    pub fn extract_all(&self) -> impl Iterator<Item = Self> {
        let s = self.clone();
        let n_elements = s.ty().num_elements().unwrap();
        (0..n_elements).map(move |i| s.extract(i))
    }
    pub fn select(&self, condition: impl Into<Self>, false_val: impl Into<Self>) -> Self {
        let condition = condition.into();
        let true_val = self;
        let false_val = false_val.into();

        assert_eq!(condition.ty(), bool::var_ty());
        assert_eq!(true_val.ty(), false_val.ty());

        let ty = true_val.ty();

        let extent = resulting_extent([&condition, true_val, &false_val].into_iter());

        new_var(
            Var {
                op: Op::KernelOp(KernelOp::Select),
                ty,
                extent,
                ..Default::default()
            },
            [&condition, true_val, &false_val],
        )
    }
    pub fn tex_lookup(&self, pos: impl Into<VarRef>) -> Self {
        let pos = pos.into();

        let elements = pos.ty().num_elements().unwrap();
        assert!(elements >= 1 && elements <= 3);

        let extent = pos.extent();

        let (_, channels) = self.extent().shape_and_channles();
        assert!(channels <= 4);
        let ty = self.ty();
        let ty = vartype::vector(ty, channels);
        let src_ref = self.get_ref();

        let composite = new_var(
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
        // assert_eq!(self.ty(), f32::var_ty());

        let shape = [
            *shape.get(0).unwrap_or(&0),
            *shape.get(1).unwrap_or(&0),
            *shape.get(2).unwrap_or(&0),
        ];
        log::trace!("Recording Texture with {shape:?}");

        let ty = self.ty();

        new_var(
            Var {
                op: Op::DeviceOp(DeviceOp::Buffer2Texture),
                ty,
                extent: Extent::Texture { shape, channels },
                ..Default::default()
            },
            [self],
        )
    }
    pub fn trace_ray(&self, ray: impl Into<Self>) -> Self {
        let ray = ray.into();

        let extent = resulting_extent([&ray]);

        assert_eq!(ray.ty(), vartype::Ray3f::var_ty());

        let ty = Intersection::var_ty();

        let accel_ref = self.get_ref();

        new_var(
            Var {
                op: Op::KernelOp(KernelOp::TraceRay),
                ty,
                extent,
                ..Default::default()
            },
            [&accel_ref, &ray],
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

        let res = new_var(
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

        let res = new_var(
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
        new_var(
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
