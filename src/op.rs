use crate::ir;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ReduceOp {
    Max,
    Min,
    Sum,
    Prod,
    Or,
    And,
    Xor,
}

/// TODO: Find better name for this kind of operation
/// Operations like creating textures and reduction operations,
/// that need their own kernels
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum DeviceOp {
    ReduceOp(ReduceOp),
    Buffer2Texture,
    BuildAccel,
}
impl DeviceOp {
    pub fn resulting_op(self) -> Op {
        match self {
            DeviceOp::ReduceOp(_) => Op::Buffer,
            DeviceOp::Buffer2Texture => Op::Texture,
            DeviceOp::BuildAccel => Op::Accel,
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
    Texture,
    Accel,
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
    pub fn evaluated(self) -> bool {
        match self {
            // Op::Nop => todo!(),
            // Op::Ref { mutable } => todo!(),
            Op::Buffer | Op::Texture { .. } | Op::Accel => true,
            _ => false,
        }
    }
    /// Gives the Operation/Variable Type, this operation should evaluate to
    pub fn resulting_op(self) -> Self {
        match self {
            Op::Buffer => Op::Buffer,
            Op::Texture { .. } => self.clone(),
            Op::Accel => Op::Accel,
            Op::KernelOp(_) => Op::Buffer,
            Op::DeviceOp(dop) => dop.resulting_op(),
            _ => todo!(),
        }
    }
}
