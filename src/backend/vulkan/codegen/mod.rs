use std::hash::Hash;

use crate::ir::IR;

mod glsl;
mod rspirv;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct DeviceInfo {
    pub work_group_size: u32,
}

pub fn assemble_trace(ir: &IR, device_info: &DeviceInfo, entry_point: &str) -> Vec<u32> {
    // rspirv::assemble_trace(ir, info, entry_point).unwrap()
    IrGlslDef {
        ir,
        entry_point,
        device_info,
    }
    .generate()
}

pub trait CodegenDef: Hash {
    fn generate(&self) -> Vec<u32>;
}

#[derive(Hash)]
pub struct IrGlslDef<'a> {
    ir: &'a IR,
    entry_point: &'a str,
    device_info: &'a DeviceInfo,
}
impl<'a> CodegenDef for IrGlslDef<'a> {
    fn generate(&self) -> Vec<u32> {
        glsl::assemble_ir(self.ir, &self.device_info, self.entry_point).unwrap()
    }
}
