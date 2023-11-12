#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bop {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// TODO: Find better name for this kind of operation
/// Operations like creating textures and reduction operations,
/// that need their own kernels
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum DeviceOp {
    Max,
}
impl DeviceOp {}

#[derive(Clone, Copy, Default, Debug, Hash, PartialEq, Eq)]
pub enum KernelOp {
    #[default]
    Nop,

    Scatter,
    Gather,
    Index,
    Literal,

    Extract(usize),
    Construct,

    Bop(Bop),

    BufferRef,
}

#[derive(Clone, Copy, Default, Debug, Hash, PartialEq, Eq)]
pub enum Op {
    #[default]
    Nop,
    // An opaque reference, causing a kernel split TODO: not sure if to call it `Opaque` or `Ref`
    // TODO: maybe split into Ref and RefMut
    Ref {
        mutable: bool,
    },
    Buffer,
    DeviceOp(DeviceOp),
    KernelOp(KernelOp),
}
impl Op {
    pub fn is_device_op(&self) -> bool {
        match self {
            Op::DeviceOp(_) => true,
            _ => false,
        }
    }
}
