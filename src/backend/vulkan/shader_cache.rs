use std::any::TypeId;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use text_placeholder::Template;

use super::codegen;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    Compute,
}
impl From<ShaderKind> for shaderc::ShaderKind {
    fn from(value: ShaderKind) -> Self {
        match value {
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
        }
    }
}

#[derive(Debug, Default)]
pub struct ShaderCache(HashMap<u64, Arc<Vec<u32>>>);

impl ShaderCache {
    pub fn get<I: codegen::CodegenDef + 'static>(&mut self, input: &I) -> Arc<Vec<u32>> {
        let mut hasher = DefaultHasher::new();
        TypeId::of::<I>().hash(&mut hasher);
        input.hash(&mut hasher);
        let hash = hasher.finish();

        self.0
            .entry(hash)
            .or_insert_with(|| Arc::new(input.generate()))
            .clone()
    }
    pub fn lease_glsl(
        &mut self,
        src: &str,
        kind: ShaderKind,
        defines: &[(&str, Option<&str>)],
    ) -> Arc<Vec<u32>> {
        log::trace!("Creating shader with the following source:");
        log::trace!("{}", src);
        log::trace!("defines: {defines:?}");
        let mut hasher = DefaultHasher::new();
        src.hash(&mut hasher);
        defines.hash(&mut hasher);
        kind.hash(&mut hasher);
        let hash = hasher.finish();

        self.0
            .entry(hash)
            .or_insert_with(|| {
                let compiler = shaderc::Compiler::new().unwrap();
                let mut options = shaderc::CompileOptions::new().unwrap();

                options.set_target_env(
                    shaderc::TargetEnv::Vulkan,
                    shaderc::EnvVersion::Vulkan1_2 as _,
                );

                for define in defines {
                    options.add_macro_definition(define.0, define.1);
                }
                let preprocessed = compiler
                    .preprocess(src, "", "main", Some(&options))
                    .unwrap()
                    .as_text();
                log::trace! {"Compiling shader: \n{preprocessed}"};

                // // For some reason I have to split this up into two parts.
                //
                // let mut options = shaderc::CompileOptions::new().unwrap();
                // options.set_target_env(
                //     shaderc::TargetEnv::Vulkan,
                //     shaderc::EnvVersion::Vulkan1_2 as _,
                // );
                //
                // let result = compiler
                //     .compile_into_spirv(&preprocessed, kind.into(), "", "main", Some(&options))
                //     .unwrap();
                // let binary = result.as_binary().to_vec();
                // Arc::new(binary)

                let compiler = glslang::Compiler::acquire().unwrap();
                let options = glslang::CompilerOptions {
                    source_language: glslang::SourceLanguage::GLSL,
                    target: glslang::Target::Vulkan {
                        version: glslang::VulkanVersion::Vulkan1_3,
                        spirv_version: glslang::SpirvVersion::SPIRV1_5,
                    },
                    ..Default::default()
                };

                let shader_stage = match kind {
                    ShaderKind::Compute => glslang::ShaderStage::Compute,
                };

                let shader = glslang::ShaderSource::from(preprocessed.as_str());
                let shader =
                    glslang::ShaderInput::new(&shader, shader_stage, &options, None).unwrap();
                let shader = compiler
                    .create_shader(shader)
                    .map_err(|err| anyhow::anyhow!("{err} {preprocessed}"))
                    .unwrap();
                let code = shader.compile().unwrap();
                Arc::new(code)
            })
            .clone()
    }
}
