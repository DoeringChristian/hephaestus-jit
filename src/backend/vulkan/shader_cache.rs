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
    pub fn lease_glsl(&mut self, src: &str, kind: ShaderKind) -> Arc<Vec<u32>> {
        let mut hasher = DefaultHasher::new();
        src.hash(&mut hasher);
        kind.hash(&mut hasher);
        let hash = hasher.finish();

        self.0
            .entry(hash)
            .or_insert_with(|| {
                let compiler = shaderc::Compiler::new().unwrap();
                let options = shaderc::CompileOptions::new().unwrap();
                let result = compiler
                    .compile_into_spirv(src, kind.into(), "", "main", Some(&options))
                    .unwrap();
                let binary = result.as_binary().to_vec();
                Arc::new(binary)
            })
            .clone()
    }
    pub fn lease_glsl_template(
        &mut self,
        template: &str,
        table: &HashMap<&str, &str>,
        kind: ShaderKind,
    ) -> Arc<Vec<u32>> {
        let templater = Template::new(template);
        self.lease_glsl(&templater.fill_with_hashmap(table), kind)
    }
}
