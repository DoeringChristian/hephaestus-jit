use crate::ir;

/// TODO: Find better name for this kind of operation
/// Operations like creating textures and reduction operations,
/// that need their own kernels
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum DeviceOp {
    Max,
    Buffer2Texture { shape: [usize; 3], channels: usize },
}
impl DeviceOp {
    pub fn resulting_op(self) -> Op {
        match self {
            DeviceOp::Max => Op::Buffer,
            DeviceOp::Buffer2Texture { shape, channels } => Op::Texture { shape, channels },
        }
    }
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
    // TODO: Split into Evaluated, DeviceOp, KernelOp?
    Buffer,
    Texture {
        shape: [usize; 3],
        channels: usize,
    },
    DeviceOp(DeviceOp),
    KernelOp(ir::Op),
}
impl Op {
    pub fn is_device_op(self) -> bool {
        match self {
            Op::DeviceOp(_) => true,
            _ => false,
        }
    }
    /// Gives the Operation/Variable Type, this operation should evaluate to
    pub fn resulting_op(self) -> Self {
        match self {
            Op::Buffer => Op::Buffer,
            Op::Texture { .. } => self.clone(),
            Op::KernelOp(_) => Op::Buffer,
            Op::DeviceOp(dop) => dop.resulting_op(),
            _ => todo!(),
        }
    }
}
