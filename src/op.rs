#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bop {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
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
    Scatter,
    Gather,
    Index,
    Literal,

    Extract(usize),
    Construct,

    Bop(Bop),
}
