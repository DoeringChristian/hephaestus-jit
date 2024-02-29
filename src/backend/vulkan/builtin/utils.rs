use ash::vk;

use crate::backend::vulkan::codegen;
use crate::backend::vulkan::vulkan_core::pipeline::{self, PipelineDef, PipelineInfo, ShaderKind};
use crate::vartype::VarType;

pub fn glsl_ty(ty: &VarType) -> &'static str {
    match ty {
        VarType::Bool => "uint8_t", // NOTE: use uint8_t for bool
        VarType::I8 => "int8_t",
        VarType::U8 => "uint8_t",
        VarType::I16 => "int16_t",
        VarType::U16 => "uint16_t",
        VarType::I32 => "int32_t",
        VarType::U32 => "uint32_t",
        VarType::I64 => "int64_t",
        VarType::U64 => "uint64_t",
        VarType::F16 => "float16_t",
        VarType::F32 => "float32_t",
        VarType::F64 => "float64_t",
        _ => todo!(),
    }
}
// TODO: imporve this
pub fn glsl_short_ty(ty: &VarType) -> &'static str {
    match ty {
        VarType::I8 => "i8",
        VarType::U8 => "u8",
        VarType::I16 => "i16",
        VarType::U16 => "u16",
        VarType::I32 => "i32",
        VarType::U32 => "u32",
        VarType::I64 => "i64",
        VarType::U64 => "u64",
        VarType::F16 => "f16",
        VarType::F32 => "f32",
        VarType::F64 => "f64",
        _ => todo!(),
    }
}
pub fn component_type(ty: &VarType) -> vk::ComponentTypeKHR {
    match ty {
        VarType::I8 => vk::ComponentTypeKHR::SINT8,
        VarType::U8 => vk::ComponentTypeKHR::UINT8,
        VarType::I16 => vk::ComponentTypeKHR::SINT16,
        VarType::U16 => vk::ComponentTypeKHR::UINT16,
        VarType::I32 => vk::ComponentTypeKHR::SINT32,
        VarType::U32 => vk::ComponentTypeKHR::UINT32,
        VarType::I64 => vk::ComponentTypeKHR::SINT64,
        VarType::U64 => vk::ComponentTypeKHR::UINT64,
        VarType::F16 => vk::ComponentTypeKHR::FLOAT16,
        VarType::F32 => vk::ComponentTypeKHR::FLOAT32,
        VarType::F64 => vk::ComponentTypeKHR::FLOAT64,
        _ => todo!(),
    }
}

#[derive(Hash)]
pub struct GlslShaderDef<'a> {
    pub code: &'a str,
    pub kind: ShaderKind,
    pub defines: &'a [(&'a str, Option<&'a str>)],
}
impl<'a> GlslShaderDef<'a> {
    pub fn compile(self) -> Box<[u32]> {
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();

        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as _,
        );

        for define in self.defines {
            options.add_macro_definition(define.0, define.1);
        }
        let preprocessed = compiler
            .preprocess(self.code, "", "main", Some(&options))
            .unwrap()
            .as_text();
        log::trace! {"Compiling shader: \n{preprocessed}"};

        let compiler = glslang::Compiler::acquire().unwrap();
        let options = glslang::CompilerOptions {
            source_language: glslang::SourceLanguage::GLSL,
            target: glslang::Target::Vulkan {
                version: glslang::VulkanVersion::Vulkan1_3,
                spirv_version: glslang::SpirvVersion::SPIRV1_5,
            },
            ..Default::default()
        };

        let shader_stage = match self.kind {
            ShaderKind::Compute => glslang::ShaderStage::Compute,
        };

        let shader = glslang::ShaderSource::from(preprocessed.as_str());
        let shader = glslang::ShaderInput::new(&shader, shader_stage, &options, None).unwrap();
        let shader = compiler
            .create_shader(shader)
            .map_err(|err| anyhow::anyhow!("{err} {preprocessed}"))
            .unwrap();
        let code = shader.compile().unwrap();
        code.into_boxed_slice()
    }
}
impl<'a> PipelineDef for GlslShaderDef<'a> {
    #[profiling::function]
    fn generate(self) -> PipelineInfo {
        self.compile().generate()
    }
}
