#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bop {
    // Normal Binary Operations
    Add,
    Sub,
    Mul,
    Div,
    Modulus,
    Min,
    Max,
    // Bitwise
    And,
    Or,
    Xor,
    // Shift
    Shl,
    Shr,

    // Comparisons
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Uop {
    // Casting
    Cast,
    BitCast,
    // Arithmetic
    Neg,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Exp2,
    Log2,
}

/// Intermediary Representation Specific Operations
#[derive(Clone, Copy, Default, Debug, Hash, PartialEq, Eq)]
pub enum KernelOp {
    #[default]
    Nop,

    Scatter,
    ScatterReduce(ReduceOp),
    ScatterAtomic(ReduceOp),
    AtomicInc,
    Gather,
    Index,
    Literal,

    Extract(u32),
    DynExtract,
    Construct,

    Select,

    LoopStart,
    LoopEnd,

    TexLookup,
    TraceRay,

    Bop(Bop),
    Uop(Uop),

    FMA,

    // Operations that are only available in IR
    BufferRef,
    TextureRef {
        dim: u32,
    }, // not sure if it's a good idea to put it here
    AccelRef,
}

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
    PrefixSum {
        inclusive: bool,
    },
    Compress, // TODO: determine if return or return by ref is better

    // Cooperative Matrix Multiply Add
    // multiply a matrix A of size [result_height]x[depth] with B of size [depth]x[result_width]
    // and add C of size [result_height]x[result_width]
    // TODO: make size variable to allow for dynamic batch sizes
    MatMul {
        max_n: usize,
        max_m: usize,
        max_k: usize,
    },
    Buffer2Texture,
    BuildAccel,
}
impl DeviceOp {
    pub fn resulting_op(self) -> Op {
        match self {
            DeviceOp::ReduceOp(_) => Op::Buffer,
            DeviceOp::Compress => Op::Nop,
            DeviceOp::PrefixSum { .. } => Op::Buffer,
            DeviceOp::MatMul { .. } => Op::Buffer,
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
    KernelOp(KernelOp),
}
impl Op {
    pub fn is_device_op(self) -> bool {
        matches!(self, Op::DeviceOp(_))
    }
    pub fn evaluated(self) -> bool {
        matches!(self, Op::Buffer | Op::Texture { .. } | Op::Accel)
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
