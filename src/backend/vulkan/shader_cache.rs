use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use text_placeholder::Template;

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

                // For some reason I have to split this up into two parts.
                let preprocessed = compiler
                    .preprocess(src, "", "main", Some(&options))
                    .unwrap()
                    .as_text();
                log::trace! {"Compiling shader: \n{preprocessed}"};

                let mut options = shaderc::CompileOptions::new().unwrap();
                options.set_target_env(
                    shaderc::TargetEnv::Vulkan,
                    shaderc::EnvVersion::Vulkan1_2 as _,
                );

                let result = compiler
                    .compile_into_spirv(&preprocessed, kind.into(), "", "main", Some(&options))
                    .unwrap();
                let binary = result.as_binary().to_vec();
                Arc::new(binary)
            })
            .clone()
    }
}
