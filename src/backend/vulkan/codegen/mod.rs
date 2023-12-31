use crate::ir::IR;

mod glsl;
mod rspirv;
mod spvasm;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CompileInfo {
    pub work_group_size: u32,
}

pub fn assemble_trace(ir: &IR, info: &CompileInfo, entry_point: &str) -> Vec<u32> {
    spvasm::assemble_ir(ir, info, entry_point).unwrap()
    // rspirv::assemble_trace(ir, info, entry_point).unwrap()
    // glsl::assemble_ir(ir, info, entry_point).unwrap()
}
