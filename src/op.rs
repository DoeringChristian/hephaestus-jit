#[derive(Clone, Copy, Default, Debug)]
pub enum Op {
    #[default]
    Nop,
    // An opaque reference, causing a kernel split TODO: not sure if to call it `Opaque` or `Ref`
    Ref,
    Buffer,
    Add,
    Scatter,
    Gather,
    Index,
    Literal,
}
