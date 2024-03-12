use std::cell::RefCell;
use std::ops::Range;

use indexmap::IndexMap;

use crate::extent::Extent;
use crate::ir::IR;
use crate::op::{DeviceOp, KernelOp, Op};
use crate::prehashed::Prehashed;
use crate::resource::{Resource, ResourceDesc};
use crate::vartype::{self, VarType};
use crate::{backend, compiler, AsVarType};

use self::backend::{AccelDesc, ArrayDesc, TextureDesc};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Var(pub(crate) usize);

#[derive(Debug)]
pub struct Entry {
    pub op: Op,
    pub ty: &'static VarType,
    pub extent: Extent,
    pub deps: Vec<Var>,
    pub literal: u64,
}
impl Entry {
    pub fn desc(&self) -> ResourceDesc {
        match &self.extent {
            Extent::Size(size) => ResourceDesc::ArrayDesc(ArrayDesc {
                size: *size,
                ty: self.ty,
            }),
            // Extent::DynSize { capacity, size } => ResourceDesc::ArrayDesc(ArrayDesc {
            //     size: capacity,
            //     ty: self.ty,
            // }),
            Extent::Texture { shape, channels } => ResourceDesc::TextureDesc(TextureDesc {
                shape: *shape,
                channels: *channels,
                format: self.ty,
            }),
            Extent::Accel(desc) => ResourceDesc::AccelDesc(desc.clone()),
            _ => todo!(),
        }
    }
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
    pub entries: Vec<Entry>,
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
    pub fn entry_mut(&mut self, var: Var) -> &mut Entry {
        &mut self.entries[var.0]
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

///
/// Returns a variable that represents a global index within a kernel, while enforcing a minimum
/// size.
///
pub fn sized_index(size: usize) -> Var {
    new_var(Entry {
        op: Op::KernelOp(KernelOp::Index),
        ty: u32::var_ty(),
        extent: Extent::Size(size),
        ..Default::default()
    })
}

///
/// Returns the [Extent], resulting from an operation on some variables.
///
fn resulting_extent<'a>(vars: impl IntoIterator<Item = Var>) -> Extent {
    with_ftrace(|trace| {
        vars.into_iter()
            .map(|var| trace.entry(var).extent.clone())
            .reduce(|a, b| a.resulting_extent(&b))
            .unwrap()
    })
}

impl Var {
    pub fn ty(&self) -> &'static VarType {
        with_ftrace(|trace| trace.entry(*self).ty)
    }
    pub fn extent(&self) -> Extent {
        with_ftrace(|trace| trace.entry(*self).extent.clone())
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
        new_var(Entry {
            op: Op::Ref { mutable },
            ty,
            extent: Extent::Size(0),
            deps: vec![*self],
            ..Default::default()
        })
    }

    pub fn scatter(self, dst: &mut Self, index: Self) {
        let ty = self.ty();
        assert_eq!(self.ty(), ty);
        let extent = resulting_extent([self, index]);

        let dst_ref = dst.get_mut();

        let scatter = new_var(Entry {
            op: Op::KernelOp(KernelOp::Scatter),
            ty: vartype::void(),
            extent,
            deps: vec![dst_ref, self, index],
            ..Default::default()
        });

        let extent = dst.extent();

        let phi = new_var(Entry {
            op: Op::Buffer,
            ty,
            extent,
            deps: vec![*dst, scatter],
            ..Default::default()
        });
        *dst = phi;
    }
}

#[cfg(test)]
mod test {
    use crate::Graph;

    use super::*;
    #[test]
    fn ftrace_01() {
        let device = backend::vulkan(0);
        let mut x = sized_literal(1, 10);

        let index = sized_index(3);
        sized_literal(3, 3).scatter(&mut x, index);

        with_ftrace(|trace| {
            dbg!(trace);
        });
        let graph = Graph::compile(&[], &[x]);
        dbg!(&graph);
        let resources = graph.launch(&device, &[]);
        dbg!(&resources);
        dbg!(resources[0]
            .buffer()
            .unwrap()
            .to_host::<i32>(0..10)
            .unwrap());
    }
}
