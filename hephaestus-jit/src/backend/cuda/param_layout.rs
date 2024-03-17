use std::collections::HashMap;

use crate::ir::{VarId, IR};
use crate::vartype::VarType;

#[derive(Clone, Debug, Default)]
pub struct ParamLayout(HashMap<VarId, usize>);

impl ParamLayout {
    pub fn byte_offset(&self, id: VarId) -> usize {
        self.0[&id] * std::mem::size_of::<u64>()
    }
    pub fn buffer_size(&self) -> usize {
        2 * 8
    }
    pub fn generate(trace: &IR) -> Self {
        todo!()
        // let mut offset: usize = 0;
        //
        // let offsets = trace
        //     .arrays
        //     .iter()
        //     .map(|id| {
        //         let o = offset;
        //         offset += 1;
        //         (*id, o)
        //     })
        //     .collect::<HashMap<_, _>>();
        //
        // Self(offsets)
    }
}
