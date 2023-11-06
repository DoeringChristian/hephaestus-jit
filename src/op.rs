#[derive(Clone, Copy, Default, Debug)]
pub enum Op {
    #[default]
    Nop,
    LoadArray,
    Add,
    Scatter,
    Gather,
    Index,
    Const,
}
