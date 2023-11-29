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
    pub fn lease_static_glsl_templated(
        &mut self,
        template: &'static str,
        table: &[(&'static str, &'static str)],
        kind: ShaderKind,
    ) -> Arc<Vec<u32>> {
        log::trace!("Leasing template shader.");
        let mut hasher = DefaultHasher::new();

        template.as_ptr().hash(&mut hasher);
        template.len().hash(&mut hasher);
        kind.hash(&mut hasher);
        for t in table.iter() {
            t.0.as_ptr().hash(&mut hasher);
            t.0.len().hash(&mut hasher);
            t.1.as_ptr().hash(&mut hasher);
            t.1.len().hash(&mut hasher);
        }

        let hash = hasher.finish();

        self.0
            .entry(hash)
            .or_insert_with(|| {
                log::trace!("    Compiling Template Kernel.");
                let template = Template::new(template);

                let table = table.into_iter().cloned().collect::<HashMap<_, _>>();
                let src = template.fill_with_hashmap(&table);

                let compiler = shaderc::Compiler::new().unwrap();
                let options = shaderc::CompileOptions::new().unwrap();
                let result = compiler
                    .compile_into_spirv(&src, kind.into(), "", "main", Some(&options))
                    .unwrap();
                let binary = result.as_binary().to_vec();
                Arc::new(binary)
            })
            .clone()
    }
    pub fn lease_glsl(&mut self, src: &str, kind: ShaderKind) -> Arc<Vec<u32>> {
        log::trace!("Creating shader with the following source:");
        log::trace!("{}", src);
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
}
