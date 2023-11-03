use std::cell::RefCell;

use crate::backend::{self, Array, Device};
use crate::trace::{Op, Trace, Var, VarId, VarType};

thread_local! {
    static KERNEL: RefCell<KernelTrace> = RefCell::new(Default::default());
}

#[derive(Default, Debug)]
struct KernelTrace {
    pub(crate) trace: Trace,
    pub(crate) arrays: Vec<Array>,
}

fn with_kernel<T, F: FnOnce(&mut KernelTrace) -> T>(f: F) -> T {
    KERNEL.with(|k| {
        let mut k = k.borrow_mut();
        f(&mut k)
    })
}

pub fn index(size: usize) -> VarRef {
    with_kernel(|k| {
        VarRef(k.trace.push_var(
            Var {
                ty: VarType::U32,
                op: Op::Index,
            },
            size,
        ))
    })
}
pub fn array(array: &Array) -> VarRef {
    with_kernel(|k| {
        k.arrays.push(array.clone());
        VarRef(k.trace.push_array(Var {
            ty: array.ty(),
            op: Op::LoadArray,
        }))
    })
}

impl From<&Array> for VarRef {
    fn from(array: &Array) -> Self {
        with_kernel(|k| {
            k.arrays.push(array.clone());
            VarRef(k.trace.push_array(Var {
                ty: array.ty(),
                op: Op::LoadArray,
            }))
        })
    }
}

#[derive(Clone, Copy)]
pub struct VarRef(VarId);

impl VarRef {
    pub fn ty(self) -> VarType {
        with_kernel(|k| k.trace.var_ty(self.0).clone())
    }
    pub fn add(self, other: Self) -> Self {
        assert_eq!(self.ty(), other.ty());
        let ty = self.ty();
        VarRef(with_kernel(|k| {
            k.trace.push_var(
                Var {
                    ty,
                    op: Op::Add {
                        lhs: self.0,
                        rhs: other.0,
                    },
                },
                0,
            )
        }))
    }
    pub fn scatter(&self, target: &Self, idx: &Self) {
        let ty = self.ty();
        with_kernel(|k| {
            k.trace.push_var(
                Var {
                    ty,
                    op: Op::Scatter {
                        dst: target.0,
                        src: self.0,
                        idx: idx.0,
                    },
                },
                0,
            );
        })
    }
    pub fn gather(&self, idx: &Self) -> Self {
        let ty = self.ty();
        VarRef(with_kernel(|k| {
            k.trace.push_var(
                Var {
                    ty,
                    op: Op::Gather {
                        src: self.0,
                        idx: idx.0,
                    },
                },
                0,
            )
        }))
    }
}
pub fn launch(device: &Device) -> backend::Result<()> {
    with_kernel(|k| device.execute_trace(&k.trace, &k.arrays))
}

#[cfg(test)]
mod test {

    use crate::backend::Device;
    use crate::trace::VarType;

    use crate::tracer as jit;

    #[test]
    fn index() {
        let device = Device::cuda(0).unwrap();
        let output = device.create_array::<u32>(10).unwrap();

        {
            let output = jit::array(&output);
            let idx = jit::index(10);

            idx.scatter(&output, &idx);
        }

        jit::launch(&device).unwrap();

        dbg!(output.to_host::<u32>().unwrap());
    }
}
