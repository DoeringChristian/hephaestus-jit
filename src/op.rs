#[derive(Clone, Copy, Default, Debug)]
pub enum Op {
    #[default]
    Nop,
    Buffer,
    Add,
    Scatter,
    Gather,
    Index,
    Literal,
}
