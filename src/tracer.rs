use crate::trace::{Trace, VarId};

pub struct Tracer {
    trace: Trace,
}

pub struct VarRef<'a> {
    id: VarId,
    r: &'a Tracer,
}

impl<'a> VarRef<'a> {}

impl Tracer {}
