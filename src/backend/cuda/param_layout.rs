use std::collections::HashMap;

use crate::trace::{Trace, VarId, VarType};

#[derive(Clone, Debug)]
pub struct ParamLayout(HashMap<VarId, usize>);

impl ParamLayout {
    pub fn byte_offset(&self, id: VarId) -> usize {
        (self.0[&id] + 1) * 8
    }
    pub fn generate(trace: &Trace) -> Self {
        let mut offsets = HashMap::default();

        let mut offset: usize = 0;

        for id in trace.var_ids() {
            if trace.var(id).ty == VarType::Array {
                offsets.insert(id, offset);
                offset += 1;
            }
        }
        Self(offsets)
    }
}
