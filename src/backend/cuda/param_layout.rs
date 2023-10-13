use std::collections::HashMap;

use crate::trace::{Trace, VarId, VarType};

#[derive(Clone, Debug, Default)]
pub struct ParamLayout(HashMap<VarId, usize>);

impl ParamLayout {
    pub fn byte_offset(&self, id: VarId) -> usize {
        (self.0[&id] + 1) * std::mem::size_of::<u64>()
    }
    pub fn buffer_size(&self) -> usize {
        (self.0.len() + 1) * 8
    }
    pub fn generate(trace: &Trace) -> Self {
        let mut offsets = HashMap::default();

        let mut offset: usize = 1; // One u64 for kernel count

        for id in trace.var_ids() {
            if trace.var(id).ty == VarType::Array {
                offsets.insert(id, offset);
                offset += 1;
            }
        }
        Self(offsets)
    }
}
