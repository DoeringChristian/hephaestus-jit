use std::collections::HashMap;

use crate::trace::{Trace, VarId, VarType};

#[derive(Clone, Debug, Default)]
pub struct ParamLayout(HashMap<VarId, (usize, usize)>);

impl ParamLayout {
    pub fn generate(trace: &Trace) -> Self {
        let mut i = 0;

        const BININGS_PER_SET: usize = 6;

        let offsets = trace
            .arrays
            .iter()
            .map(|id| {
                let set = i / BININGS_PER_SET;
                let binding = i % BININGS_PER_SET;
                i += 1;
                (*id, (set, binding))
            })
            .collect::<HashMap<_, _>>();

        Self(offsets)
    }
}
