use std::collections::HashMap;

use crate::trace::{Trace, VarId, VarType};

#[derive(Clone, Debug, Default)]
pub struct ParamLayout(HashMap<VarId, usize>, usize);

impl ParamLayout {
    pub fn buffer_idx(&self, id: VarId) -> usize {
        self.0[&id]
    }
    pub fn len(&self) -> usize {
        self.1
    }
    pub fn generate(trace: &Trace) -> Self {
        let mut i = 0;

        const BININGS_PER_SET: usize = 6;

        let buffer_idxs = trace
            .arrays
            .iter()
            .map(|id| {
                let buffer_idx = i;
                // let set = i / BININGS_PER_SET;
                // let binding = i % BININGS_PER_SET;
                i += 1;
                (*id, buffer_idx)
            })
            .collect::<HashMap<_, _>>();

        Self(buffer_idxs, i)
    }
}
