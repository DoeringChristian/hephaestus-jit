use std::hash::Hash;

use crate::prehashed::Prehashed;
use ash::vk;

use crate::ir::IR;

use super::vulkan_core::pipeline::{Binding, DescSetLayout, PipelineDef, PipelineInfo};

mod glsl;
// mod rspirv;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct DeviceInfo {
    pub work_group_size: u32,
}

// #[profiling::function]
// pub fn assemble_trace(ir: &IR, device_info: &DeviceInfo, entry_point: &str) -> Vec<u32> {
//     // rspirv::assemble_trace(ir, info, entry_point).unwrap()
//     // IrGlslDef {
//     //     ir,
//     //     entry_point,
//     //     device_info,
//     // }
//     // .generate()
// }

// pub trait CodegenDef: Hash {
//     fn generate(&self) -> Vec<u32>;
// }

pub struct IrGlslDef<'a> {
    pub ir: &'a Prehashed<IR>,
    pub entry_point: &'a str,
    pub device_info: &'a DeviceInfo,
}
// Implement hash with type id
impl<'a> Hash for IrGlslDef<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::TypeId::of::<IrGlslDef<'static>>().hash(state);
        self.ir.hash(state);
        self.entry_point.hash(state);
        self.device_info.hash(state);
    }
}
impl<'a> PipelineDef for IrGlslDef<'a> {
    fn generate(self) -> PipelineInfo {
        let layouts = [DescSetLayout {
            bindings: vec![
                Binding {
                    binding: 0,
                    count: self.ir.n_buffers as u32 + 1,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                },
                Binding {
                    binding: 1,
                    count: self.ir.n_textures as u32,
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                },
                Binding {
                    binding: 2,
                    count: self.ir.n_accels as u32,
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                },
            ],
        }];
        let code = glsl::assemble_ir(self.ir, &self.device_info, self.entry_point)
            .unwrap()
            .into_boxed_slice();
        PipelineInfo {
            code,
            desc_set_layouts: layouts.into(),
        }
    }

    fn typed_hash(&self, state: &mut impl std::hash::Hasher) {
        std::any::TypeId::of::<IrGlslDef<'static>>().hash(state);
        self.hash(state);
    }
}
