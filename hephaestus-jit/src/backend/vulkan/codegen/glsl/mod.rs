use half::f16;

use super::DeviceInfo;
use crate::ir::{VarId, IR};
use crate::vartype::{self, AsVarType, VarType};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Write};

pub fn assemble_ir(ir: &IR, info: &DeviceInfo, entry_point: &str) -> Option<Vec<u32>> {
    let mut s = String::new();
    assemble_entry_point(&mut s, ir, info, entry_point).unwrap();

    log::trace!("\n{s}");

    // // Compile with glslang
    // let compiler = glslang::Compiler::acquire().unwrap();
    // let options = glslang::CompilerOptions {
    //     source_language: glslang::SourceLanguage::GLSL,
    //     target: glslang::Target::Vulkan {
    //         version: glslang::VulkanVersion::Vulkan1_3,
    //         spirv_version: glslang::SpirvVersion::SPIRV1_6,
    //     },
    //     ..Default::default()
    // };
    //
    // let shader = glslang::ShaderSource::from(s.as_str());
    // let shader =
    //     glslang::ShaderInput::new(&shader, glslang::ShaderStage::Compute, &options, None).unwrap();
    // let shader = compiler.create_shader(shader).unwrap();
    // let code = shader.compile().unwrap();

    // Compile with shaderc
    {
        profiling::scope!("Compiling with shaderc");
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        options.set_hlsl_offsets(true);
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as _,
        );
        options.set_target_spirv(shaderc::SpirvVersion::V1_5);
        let compiler = shaderc::Compiler::new().unwrap();

        let artefact = compiler
            .compile_into_spirv(
                &s,
                shaderc::ShaderKind::Compute,
                "",
                entry_point,
                Some(&options),
            )
            .map_err(|err| anyhow::anyhow!("{err}: {s}"))
            .unwrap();
        let code = artefact.as_binary().to_vec();

        Some(code)
    }
}

#[profiling::function]
pub fn assemble_entry_point(
    s: &mut String,
    ir: &IR,
    info: &DeviceInfo,
    entry_point: &str,
) -> std::fmt::Result {
    let mut b = GlslBuilder::new(ir);
    write!(
        s,
        r#"
#version 460

// Explicit arythmetic types
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require

// Atomics and Memory
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_atomic_int64: require
#extension GL_EXT_shader_atomic_float: require

// Ray Tracing
#extension GL_EXT_ray_tracing: enable
#extension GL_EXT_ray_query: enable

// Scalar Block layout 
#extension GL_EXT_scalar_block_layout: require

// Subgroup
#extension GL_KHR_shader_subgroup_ballot: enable
#extension GL_KHR_shader_subgroup_basic: enable

// Cooperative Matrix
// #extension GL_KHR_cooperative_matrix: enable
        
"#
    )?;

    // Generate Composite types
    for var in ir.vars.iter() {
        b.composite_type(s, ir, var.ty)?;
        b.composite_type(s, ir, memory_type(var.ty))?;
    }

    // Generate Bindigns
    b.bindings(s, ir)?;
    let ty = u32::var_ty();
    b.buffer_binding(s, u32::var_ty())?;

    writeln!(s, "")?;
    writeln!(
        s,
        "layout(local_size_x = {workgroup_size}, local_size_y = 1, local_size_z = 1)in;",
        workgroup_size = info.work_group_size
    )?;
    writeln!(s, "void {entry_point}(){{")?;

    if b.ray_query {
        writeln!(s, "\trayQueryEXT ray_query;")?;
    }

    writeln!(s, "\tuint index = uint(gl_GlobalInvocationID.x);")?;

    // Early return
    writeln!(s, "\tuint size = buffer_uint32_t[0].b[0];")?;
    writeln!(s, "\tif (index >= size) {{return;}}")?;

    assemble_vars(s, ir)?;

    writeln!(s, "}}")?;

    Ok(())
}

#[derive(Debug, Default)]
pub struct GlslBuilder {
    n_buffers: usize,
    buffer_types: HashSet<&'static VarType>,
    samplers: HashSet<(u32, &'static VarType)>,
    composite_types: HashSet<&'static VarType>,
    ray_query: bool,
}
impl GlslBuilder {
    pub fn new(ir: &IR) -> Self {
        Self {
            n_buffers: ir.n_buffers + 1,
            ..Default::default()
        }
    }
    pub fn composite_type(
        &mut self,
        s: &mut String,
        ir: &IR,
        ty: &'static VarType,
    ) -> std::fmt::Result {
        if !self.composite_types.contains(&ty) {
            match ty {
                VarType::Struct { tys } => {
                    for ty in tys {
                        self.composite_type(s, ir, ty)?;
                    }
                    let ty_name = GlslTypeName(ty);
                    writeln!(s, "struct {ty_name}{{")?;
                    for (i, ty) in tys.iter().enumerate() {
                        writeln!(s, "\t{ty} e{i};", ty = GlslTypeName(ty))?;
                    }
                    writeln!(s, "}};")?;
                }
                VarType::Array {
                    ty: array_type,
                    num,
                } => {
                    let array_name = GlslTypeName(ty);
                    let internal_name = GlslTypeName(array_type);
                    writeln!(s, "#define {array_name} {internal_name}[{num}]")?;
                }
                _ => {}
            }
            self.composite_types.insert(ty);
        }
        Ok(())
    }
    pub fn buffer_binding(&mut self, s: &mut String, ty: &'static VarType) -> std::fmt::Result {
        let n_buffers = self.n_buffers;
        if !self.buffer_types.contains(ty) {
            let glsl_ty = GlslTypeName(ty);
            writeln!(s,"layout(set = 0, binding = 0, std430) buffer Buffer_{glsl_ty}{{ {glsl_ty} b[]; }} buffer_{glsl_ty}[{n_buffers}];")?;
            self.buffer_types.insert(ty);
        }
        Ok(())
    }
    pub fn sampler_binding(
        &mut self,
        s: &mut String,
        ir: &IR,
        dim: u32,
        ty: &'static VarType,
    ) -> std::fmt::Result {
        let n_samplers = ir.n_textures;
        if !self.samplers.contains(&(dim, ty)) {
            let type_prefix = match ty {
                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => "i",
                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => "u",
                VarType::F16 | VarType::F32 | VarType::F64 => "",
                _ => todo!(),
            };
            let glsl_type = GlslTypeName(ty);
            writeln!(
                s,
                "layout(set = 0, binding = 1) uniform {type_prefix}sampler{dim}D samplers{dim}D{glsl_type}[{n_samplers}];"
            )?;
            // writeln!(
            //     s,
            //     "layout(set = 0, binding = 1, rgba32f) uniform readonly {type_prefix}image{dim}D images[{n_samplers}];"
            // )?;
            self.samplers.insert((dim, ty));
        }
        Ok(())
    }
    pub fn bindings(&mut self, s: &mut String, ir: &IR) -> std::fmt::Result {
        for id in ir.var_ids() {
            let var = ir.var(id);
            match var.op {
                crate::op::KernelOp::BufferRef => {
                    let memory_type = memory_type(var.ty);
                    self.buffer_binding(s, memory_type)?;
                }
                crate::op::KernelOp::TextureRef { dim } => {
                    self.sampler_binding(s, ir, dim, var.ty)?;
                }
                crate::op::KernelOp::AccelRef => {
                    let n_accels = ir.n_accels;
                    if !self.ray_query {
                        writeln!(s, "layout(set = 0, binding = 2) uniform accelerationStructureEXT accels[{n_accels}];")?;
                        // writeln!(
                        //     s,
                        //     "layout(set = 0, binding = 2) uniform accelerationStructureEXT accels;"
                        // )?;
                        self.ray_query = true;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn memory_type(ty: &VarType) -> &VarType {
    match ty {
        VarType::Bool => u8::var_ty(),
        VarType::Vec { ty, num } if *num == 3 => vartype::array(ty, 3),
        VarType::Struct { tys } => {
            let tys = tys.iter().map(|ty| memory_type(ty)).collect::<Vec<_>>();
            vartype::composite(&tys)
        }
        VarType::Array { ty, num } => {
            let ty = memory_type(ty);
            vartype::array(&ty, *num)
        }
        _ => ty,
    }
}

pub struct GlslTypeName(&'static VarType);
impl std::fmt::Display for GlslTypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            VarType::Void => write!(f, "void"),
            VarType::Bool => write!(f, "bool"),
            VarType::I8 => write!(f, "int8_t"),
            VarType::U8 => write!(f, "uint8_t"),
            VarType::I16 => write!(f, "int16_t"),
            VarType::U16 => write!(f, "uint16_t"),
            VarType::I32 => write!(f, "int32_t"),
            VarType::U32 => write!(f, "uint32_t"),
            VarType::I64 => write!(f, "int64_t"),
            VarType::U64 => write!(f, "uint64_t"),
            VarType::F16 => write!(f, "float16_t"),
            VarType::F32 => write!(f, "float32_t"),
            VarType::F64 => write!(f, "float64_t"),
            VarType::Vec { ty, num } => match ty {
                VarType::I8 => write!(f, "i8vec{num}"),
                VarType::U8 => write!(f, "u8vec{num}"),
                VarType::I16 => write!(f, "i16{num}"),
                VarType::U16 => write!(f, "u16vec{num}"),
                VarType::I32 => write!(f, "i32vec{num}"),
                VarType::U32 => write!(f, "u32vec{num}"),
                VarType::I64 => write!(f, "i64vec{num}"),
                VarType::U64 => write!(f, "u64vec{num}"),
                VarType::F16 => write!(f, "f16vec{num}"),
                VarType::F32 => write!(f, "f32vec{num}"),
                VarType::F64 => write!(f, "f64vec{num}"),
                _ => todo!(),
            },
            VarType::Array { ty, num } => {
                let ty = GlslTypeName(ty);
                write!(f, "{ty}x{num}")
            }
            VarType::Mat { ty, rows, cols } => {
                let ty = match ty {
                    VarType::F16 => "f16",
                    VarType::F32 => "f32",
                    VarType::F64 => "f64",
                    _ => todo!(),
                };
                write!(f, "{ty}mat{rows}x{cols}")
            }
            VarType::Struct { tys } => {
                write!(f, "struct_")?;
                for ty in tys {
                    write!(f, "{}_", GlslTypeName(ty))?;
                }
                // Struct end marker is "__"
                write!(f, "end")?;
                Ok(())
            }
        }
    }
}

pub struct Reg(VarId);
impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{id}", id = self.0 .0)
    }
}

fn assemble_vars(s: &mut String, ir: &IR) -> std::fmt::Result {
    for id in ir.var_ids() {
        let var = ir.var(id);
        let deps = ir.deps(id);

        let ty = var.ty;
        let glsl_ty = GlslTypeName(ty);
        // let ty = GlslTypeName(var.ty);

        writeln!(s, "")?;
        // writeln!(
        //     s,
        //     "\t// var id={id}, scope={scope} op={op:?}",
        //     id = id.0,
        //     scope = var.scope.0,
        //     op = var.op,
        // )?;
        writeln!(s, "\t// var id={id}, op={op:?}", id = id.0, op = var.op,)?;

        match var.op {
            crate::op::KernelOp::Nop => {
                writeln!(
                    s,
                    "\t{glsl_ty} {dst} = {src};",
                    dst = Reg(id),
                    src = Reg(deps[0])
                )?;
            }
            crate::op::KernelOp::Scatter => {
                let dst = deps[0];
                let src = deps[1];
                let idx = deps[2];
                let cond = deps.get(3);

                let ty = ir.var(src).ty;
                let glsl_ty = GlslTypeName(ty);

                let buffer_idx = ir.var(dst).data + 1;

                let src = Reg(src);
                let idx = Reg(idx);
                let dst = Reg(id);

                // Forward declare `dst`
                if let Some(cond) = cond {
                    writeln!(s, "\tif ({cond}){{", cond = Reg(*cond))?;
                }
                let memory_type = memory_type(ty);
                if memory_type == ty {
                    writeln!(s, "\tbuffer_{glsl_ty}[{buffer_idx}].b[{idx}] = {src};",)?;
                } else {
                    write!(
                        s,
                        "\tbuffer_{memory_type}[{buffer_idx}].b[{idx}] = ",
                        memory_type = GlslTypeName(memory_type)
                    )?;
                    assemble_cast_inline(s, format!("{src}"), memory_type, ty)?;
                    writeln!(s, ";")?;
                }

                if let Some(_) = cond {
                    writeln!(s, "\t}}")?;
                }
            }
            crate::op::KernelOp::ScatterReduce(op) => {
                let dst = deps[0];
                let src = deps[1];
                let idx = deps[2];
                let cond = deps.get(3);

                let ty = ir.var(src).ty;
                let glsl_ty = GlslTypeName(ty);

                let buffer_idx = ir.var(dst).data + 1;

                let src = Reg(src);
                let idx = Reg(idx);
                let dst = Reg(id);

                if let Some(cond) = cond {
                    writeln!(s, "\tif({cond}){{", cond = Reg(*cond))?;
                }
                let atomic_fn = match op {
                    crate::op::ReduceOp::Max => "atomicMax",
                    crate::op::ReduceOp::Min => "atomicMin",
                    crate::op::ReduceOp::Sum => "atomicAdd",
                    crate::op::ReduceOp::Prod => todo!(),
                    crate::op::ReduceOp::Or => "atomicOr",
                    crate::op::ReduceOp::And => "atomicAnd",
                    crate::op::ReduceOp::Xor => "atomicXor",
                };

                let memory_type = memory_type(ty);
                if memory_type == ty {
                    writeln!(
                        s,
                        "\t{atomic_fn}(buffer_{glsl_ty}[{buffer_idx}].b[{idx}], {src});"
                    )?;
                } else {
                    write!(
                        s,
                        "\t{atomic_fn}(buffer_{memory_type}[{buffer_idx}].b[{idx}], ",
                        memory_type = GlslTypeName(memory_type),
                    )?;
                    assemble_cast_inline(s, format!("{src}"), memory_type, ty)?;
                    writeln!(s, ");")?;
                }
                if let Some(_) = cond {
                    writeln!(s, "\t}}")?;
                }
            }
            crate::op::KernelOp::ScatterAtomic(op) => {
                let dst = deps[0];
                let src = deps[1];
                let idx = deps[2];
                let cond = deps.get(3);

                let ty = ir.var(src).ty;
                let glsl_ty = GlslTypeName(ty);

                let buffer_idx = ir.var(dst).data + 1;

                let src = Reg(src);
                let idx = Reg(idx);
                let dst = Reg(id);

                writeln!(s, "\t{glsl_ty} {dst};")?;
                if let Some(cond) = cond {
                    writeln!(s, "\tif({cond}){{", cond = Reg(*cond))?;
                } else {
                    writeln!(s, "\t{{")?;
                }
                let atomic_fn = match op {
                    crate::op::ReduceOp::Max => "atomicMax",
                    crate::op::ReduceOp::Min => "atomicMin",
                    crate::op::ReduceOp::Sum => "atomicAdd",
                    crate::op::ReduceOp::Prod => todo!(),
                    crate::op::ReduceOp::Or => "atomicOr",
                    crate::op::ReduceOp::And => "atomicAnd",
                    crate::op::ReduceOp::Xor => "atomicXor",
                };
                let memory_type = memory_type(ty);
                if memory_type == ty {
                    writeln!(
                        s,
                        "\t{dst} = {atomic_fn}(buffer_{glsl_ty}[{buffer_idx}].b[{idx}], {src});"
                    )?;
                } else {
                    write!(s, "\t{memory_type} tmp = {atomic_fn}(buffer_{memory_type}[{buffer_idx}].b[{idx}], ", memory_type = GlslTypeName(memory_type))?;
                    assemble_cast_inline(s, format!("{src}"), memory_type, ty)?;
                    writeln!(s, ");")?;
                    write!(s, "\t{dst} = ")?;
                    assemble_cast_inline(s, "tmp", ty, memory_type)?;
                    writeln!(s, ");")?;
                }
                writeln!(s, "\t}}")?;
            }
            crate::op::KernelOp::AtomicInc => {
                let dst = deps[0];
                let idx = deps[1];
                let cond = deps[2];
                let buffer_idx = ir.var(dst).data + 1;

                let dst = Reg(id);
                let idx = Reg(idx);

                let cond = Reg(cond);
                writeln!(s, "\t{glsl_ty} {dst};")?;
                writeln!(s, "\t{{")?;
                writeln!(s, "\t\tuvec4 activemask = subgroupBallot({cond});")?;
                writeln!(
                    s,
                    "\t\tuint activelanes = subgroupBallotBitCount(activemask);"
                )?;
                writeln!(s, "\t\tuint laneid = gl_SubgroupInvocationID;")?;
                writeln!(s, "\t\tuvec4 ltmask = gl_SubgroupLtMask;")?;
                writeln!(s, "\t\tuint32_t warpid;")?;
                writeln!(s, "\t\tif (laneid == 0){{")?;
                writeln!(s, "\t\t\twarpid = atomicAdd(buffer_{glsl_ty}[{buffer_idx}].b[{idx}], activelanes);")?;
                writeln!(s, "\t\t}}")?;
                writeln!(s, "\t\twarpid = subgroupBroadcast(warpid, 0);")?;
                writeln!(
                    s,
                    "\t\t{dst} = warpid + subgroupBallotBitCount(activemask & ltmask);"
                )?;
                writeln!(s, "\t}}")?;
            }
            crate::op::KernelOp::Gather => {
                let src = deps[0];
                let idx = deps[1];
                let cond = deps.get(2);
                let buffer_idx = ir.var(src).data + 1;

                let dst = Reg(id);
                let idx = Reg(idx);

                let memory_type = memory_type(ty);

                if let Some(cond) = cond {
                    let cond = Reg(*cond);
                    if zeroable(ty) {
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}(0);", dst = Reg(id))?;
                    } else {
                        writeln!(s, "\t{glsl_ty} {dst};", dst = Reg(id))?;
                    }
                    if memory_type == ty {
                        writeln!(
                            s,
                            "\tif ({cond}) {{ {dst} = buffer_{glsl_ty}[{buffer_idx}].b[{idx}]; }}"
                        )?;
                    } else {
                        writeln!(s, "\tif({cond}) {{")?;
                        writeln!(
                            s,
                            "\t\t{memory_type} tmp = buffer_{memory_type}[{buffer_idx}].b[{idx}];",
                            memory_type = GlslTypeName(memory_type)
                        )?;
                        write!(s, "\t\t{dst} = ")?;
                        assemble_cast_inline(s, "tmp", ty, memory_type)?;
                        writeln!(s, ";")?;
                        writeln!(s, "\t}}")?;
                    }
                } else {
                    if memory_type == ty {
                        writeln!(
                            s,
                            "\t{glsl_ty} {dst} = buffer_{glsl_ty}[{buffer_idx}].b[{idx}];",
                        )?;
                    } else {
                        writeln!(s, "\t{glsl_ty} {dst};")?;
                        writeln!(s, "\t{{")?;
                        writeln!(
                            s,
                            "\t\t{memory_type} tmp = buffer_{memory_type}[{buffer_idx}].b[{idx}];",
                            memory_type = GlslTypeName(memory_type)
                        )?;

                        write!(s, "\t\t{dst} = ")?;
                        assemble_cast_inline(s, "tmp", ty, memory_type)?;
                        writeln!(s, ";")?;
                        writeln!(s, "\t}}")?;
                    }
                }
            }
            crate::op::KernelOp::Index => {
                writeln!(s, "\tuint32_t {dst} = index;", dst = Reg(id))?;
            }
            crate::op::KernelOp::Literal => {
                let dst = Reg(id);
                match ty {
                    VarType::Bool => {
                        writeln!(
                            s,
                            "\t{glsl_ty} {dst} = {glsl_ty}({data});",
                            data = if var.data == 0 { false } else { true }
                        )?;
                    }
                    VarType::I8
                    | VarType::U8
                    | VarType::I16
                    | VarType::U16
                    | VarType::I32
                    | VarType::U32
                    | VarType::I64
                    | VarType::U64 => {
                        let data = var.data;
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data}ul);",)?;
                    }
                    VarType::F16 => {
                        let data: f16 = unsafe { *(&var.data as *const _ as *const _) };
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data});",)?;
                    }
                    VarType::F32 => {
                        let data: f32 = unsafe { *(&var.data as *const _ as *const _) };
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data});",)?;
                    }
                    VarType::F64 => {
                        let data: f64 = unsafe { *(&var.data as *const _ as *const _) };
                        writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({data});",)?;
                    }
                    _ => todo!(),
                }
            }
            crate::op::KernelOp::Extract(elem) => {
                let src = Reg(deps[0]);
                let dst = Reg(id);
                let src_ty = ir.var(deps[0]).ty;
                match src_ty {
                    VarType::Vec { ty, num } => {
                        let swizzle = match elem {
                            0 => "x",
                            1 => "y",
                            2 => "z",
                            3 => "w",
                            _ => todo!(),
                        };
                        writeln!(s, "\t{glsl_ty} {dst} = {src}.{swizzle};")
                    }
                    VarType::Array { ty, num } => writeln!(s, "\t{glsl_ty} {dst} = {src}[{elem}];"),
                    VarType::Struct { tys } => writeln!(s, "\t{glsl_ty} {dst} = {src}.e{elem};"),
                    _ => todo!(),
                }?;
            }
            crate::op::KernelOp::DynExtract => {
                let src = Reg(deps[0]);
                let idx = Reg(deps[1]);
                let dst = Reg(id);
                let src_ty = ir.var(deps[0]).ty;
                match src_ty {
                    VarType::Array { ty, num } => {
                        writeln!(s, "\t{glsl_ty} {dst} = {src}[{idx}];")?;
                    }
                    _ => todo!(),
                }
            }
            crate::op::KernelOp::Construct => {
                let dst = Reg(id);
                match ty {
                    VarType::Vec { ty, num } => {
                        write!(s, "\t{glsl_ty} {dst} = {glsl_ty}(")?;
                        for (i, id) in deps.iter().enumerate() {
                            let src = Reg(*id);
                            if i == deps.len() - 1 {
                                write!(s, "{src}")?;
                            } else {
                                write!(s, "{src},")?;
                            }
                        }
                        write!(s, "\t);\n")?;
                    }
                    VarType::Struct { .. } | VarType::Array { .. } => {
                        write!(s, "\t{glsl_ty} {dst} = {glsl_ty}(")?;
                        for (i, id) in deps.iter().enumerate() {
                            let src = Reg(*id);
                            if i == deps.len() - 1 {
                                write!(s, "{src}")?;
                            } else {
                                write!(s, "{src},")?;
                            }
                        }
                        write!(s, ");\n")?;
                    }
                    VarType::Mat { ty, cols, rows } => {
                        writeln!(s, "\t{glsl_ty} {dst};")?;
                        for (i, id) in deps.iter().enumerate() {
                            let src = Reg(*id);
                            writeln!(s, "\t{dst}[{i}] = {src};")?;
                        }
                    }
                    _ => todo!(),
                };
            }
            crate::op::KernelOp::Select => {
                let dst = Reg(id);
                let cond = Reg(deps[0]);
                let true_val = Reg(deps[1]);
                let false_val = Reg(deps[2]);
                writeln!(s, "\t{glsl_ty} {dst} = {cond} ? {true_val} : {false_val};")?;
            }
            crate::op::KernelOp::LoopStart => {
                let loop_state = Reg(id);
                let init = Reg(deps[0]);
                writeln!(s, "\t{glsl_ty} {loop_state} = {init};")?;
                writeln!(s, "\twhile({loop_state}.e0){{")?
            }
            crate::op::KernelOp::LoopEnd => {
                let dst = Reg(id);
                let loop_state = Reg(deps[0]);
                let iresult = Reg(deps[1]);

                writeln!(s, "\t{loop_state} = {iresult};")?;
                writeln!(s, "\t}}")?;
                writeln!(s, "\t{glsl_ty} {dst} = {loop_state};")?;
            }
            crate::op::KernelOp::IfStart => {
                let if_state = Reg(id);
                let init = Reg(deps[0]);
                writeln!(s, "\t{glsl_ty} {if_state} = {init};")?;
                writeln!(s, "\tif({if_state}.e0){{")?;
            }
            crate::op::KernelOp::IfEnd => {
                let dst = Reg(id);
                let if_state = Reg(deps[0]);
                let iresult = Reg(deps[1]);

                writeln!(s, "\t{if_state} = {iresult};")?;
                writeln!(s, "\t}}")?;
                writeln!(s, "\t{glsl_ty} {dst} = {if_state};")?;
            }
            crate::op::KernelOp::TexLookup => {
                let tex_ref = ir.var(deps[0]);
                let dim = match tex_ref.op {
                    crate::op::KernelOp::TextureRef { dim } => dim,
                    _ => todo!(),
                };
                let tex_type = GlslTypeName(tex_ref.ty);

                let sampler_idx = tex_ref.data;
                let coord = Reg(deps[1]);

                let dst = Reg(id);

                writeln!(
                    s,
                    "\t{glsl_ty} {dst} = texture(samplers{dim}D{tex_type}[{sampler_idx}], {coord});"
                )?;
            }
            crate::op::KernelOp::TraceRay => {
                let accel_idx = ir.var(deps[0]).data;
                let ray = Reg(deps[1]);
                // let o = Reg(deps[1]);
                // let d = Reg(deps[2]);
                // let tmin = Reg(deps[3]);
                // let tmax = Reg(deps[4]);

                let dst = Reg(id);

                writeln!(s, "\trayQueryInitializeEXT(ray_query, accels[{accel_idx}], gl_RayFlagsOpaqueEXT, 0xff, {ray}.e0, {ray}.e2, {ray}.e1, {ray}.e3);")?;
                // writeln!(s, "\trayQueryInitializeEXT(ray_query, accels[{accel_idx}], gl_RayFlagsOpaqueEXT, 0xff, {o}, {tmin}, {d}, {tmax});")?;
                // writeln!(s, "\trayQueryInitializeEXT(ray_query, accels[{accel_idx}], gl_RayFlagsOpaqueEXT, 0xff, vec3(0.6, 0.6, 1.0), 0.001, vec3(0, 0, -1), 10000.);")?;
                // writeln!(s, "\trayQueryInitializeEXT(ray_query, accels[{accel_idx}], 0x01, 0xff, vec3(0.6, 0.6, 0.1), -10, vec3(0, 0, -1), 10000.);")?;

                write!(
                    s,
                    r#"
                    {glsl_ty} {dst};
                    {dst}.e3 = 0;
                    while (rayQueryProceedEXT(ray_query)){{
                    
                        uint32_t intersection_ty = rayQueryGetIntersectionTypeEXT(ray_query, false);
                        bool opaque = intersection_ty == 0;
                        if (opaque) rayQueryConfirmIntersectionEXT(ray_query);
                    
                        // if (intersection_ty == gl_RayQueryCandidateIntersectionTriangleEXT){{
                        //     rayQueryConfirmIntersectionEXT(ray_query);
                        // }}
                    }}
                    {{
                        vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);
                        // {dst}.e0 = 1.;
                        {dst}.e0 = barycentrics.x;
                        {dst}.e1 = barycentrics.y;
                        {dst}.e2 = rayQueryGetIntersectionInstanceIdEXT(ray_query, true);
                        {dst}.e3 = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
                        {dst}.e4 = rayQueryGetIntersectionTypeEXT(ray_query, true);
                    }}
                "#
                )?;
                writeln!(s, "")?;
            }
            crate::op::KernelOp::Bop(op) => {
                let dst = Reg(id);
                let lhs = Reg(deps[0]);
                let rhs = Reg(deps[1]);
                match op {
                    // Arithmetic
                    crate::op::Bop::Add => writeln!(s, "\t{glsl_ty} {dst} = {lhs} + {rhs};")?,
                    crate::op::Bop::Sub => writeln!(s, "\t{glsl_ty} {dst} = {lhs} - {rhs};")?,
                    crate::op::Bop::Mul => writeln!(s, "\t{glsl_ty} {dst} = {lhs} * {rhs};")?,
                    crate::op::Bop::Div => writeln!(s, "\t{glsl_ty} {dst} = {lhs} / {rhs};")?,
                    crate::op::Bop::Modulus => writeln!(s, "\t{glsl_ty} {dst} = {lhs} % {rhs};")?,
                    crate::op::Bop::Min => writeln!(s, "\t{glsl_ty} {dst} = min({lhs}, {rhs});")?,
                    crate::op::Bop::Max => writeln!(s, "\t{glsl_ty} {dst} = max({lhs}, {rhs});")?,
                    // Vector
                    crate::op::Bop::Dot => writeln!(s, "\t{glsl_ty} {dst} = dot({lhs}, {rhs});")?,
                    // Binary
                    crate::op::Bop::And => match ty {
                        VarType::Bool => writeln!(s, "\t{glsl_ty} {dst} = {lhs} && {rhs};")?,
                        _ => writeln!(s, "\t{glsl_ty} {dst} = {lhs} & {rhs};")?,
                    },
                    crate::op::Bop::Or => match ty {
                        VarType::Bool => writeln!(s, "\t{glsl_ty} {dst} = {lhs} || {rhs};")?,
                        _ => writeln!(s, "\t{glsl_ty} {dst} = {lhs} | {rhs};")?,
                    },
                    crate::op::Bop::Xor => match ty {
                        VarType::Bool => writeln!(s, "\t{glsl_ty} {dst} = {lhs} != {rhs};")?,
                        _ => writeln!(s, "\t{glsl_ty} {dst} = {lhs} ^ {rhs};")?,
                    },
                    crate::op::Bop::Shl => writeln!(s, "\t{glsl_ty} {dst} = {lhs} << {rhs};")?,
                    crate::op::Bop::Shr => writeln!(s, "\t{glsl_ty} {dst} = {lhs} >> {rhs};")?,
                    //Comparison
                    crate::op::Bop::Eq => writeln!(s, "\t{glsl_ty} {dst} = {lhs} == {rhs};")?,
                    crate::op::Bop::Neq => writeln!(s, "\t{glsl_ty} {dst} = {lhs} != {rhs};")?,
                    crate::op::Bop::Lt => writeln!(s, "\t{glsl_ty} {dst} = {lhs} < {rhs};")?,
                    crate::op::Bop::Le => writeln!(s, "\t{glsl_ty} {dst} = {lhs} <= {rhs};")?,
                    crate::op::Bop::Gt => writeln!(s, "\t{glsl_ty} {dst} = {lhs} > {rhs};")?,
                    crate::op::Bop::Ge => writeln!(s, "\t{glsl_ty} {dst} = {lhs} >= {rhs};")?,
                };
            }
            crate::op::KernelOp::Uop(op) => {
                let dst = Reg(id);
                let src = Reg(deps[0]);
                let src_ty = ir.var(deps[0]).ty;
                match op {
                    crate::op::Uop::Cast => assemble_cast(s, Reg(id), Reg(deps[0]), ty, src_ty),
                    crate::op::Uop::BitCast => match (ty, src_ty) {
                        (VarType::F16, VarType::I16) => {
                            writeln!(s, "\t{glsl_ty} {dst} = int16BitsToHalf({src});")
                        }
                        (VarType::F16, VarType::U16) => {
                            writeln!(s, "\t{glsl_ty} {dst} = uint16BitsToHalf({src});")
                        }
                        (VarType::F32, VarType::I32) => {
                            writeln!(s, "\t{glsl_ty} {dst} = int32BitsToFloat({src});")
                        }
                        (VarType::F32, VarType::U32) => {
                            writeln!(s, "\t{glsl_ty} {dst} = uint32BitsToFloat({src});")
                        }
                        (VarType::F64, VarType::I64) => {
                            writeln!(s, "\t{glsl_ty} {dst} = int64BitsToDouble({src});")
                        }
                        (VarType::F64, VarType::U64) => {
                            writeln!(s, "\t{glsl_ty} {dst} = uint64BitsToDouble({src});")
                        }
                        (VarType::I16, VarType::F16) => {
                            writeln!(s, "\t{glsl_ty} {dst} = halfBitsToInt16({src});")
                        }
                        (VarType::U16, VarType::F16) => {
                            writeln!(s, "\t{glsl_ty} {dst} = floatBitsToUint16({src});")
                        }
                        (VarType::I32, VarType::F32) => {
                            writeln!(s, "\t{glsl_ty} {dst} = floatBitsToInt32({src});")
                        }
                        (VarType::U32, VarType::F32) => {
                            writeln!(s, "\t{glsl_ty} {dst} = floatBitsToUint32({src});")
                        }
                        (VarType::I64, VarType::F64) => {
                            writeln!(s, "\t{glsl_ty} {dst} = doubleBitsToInt64({src});")
                        }
                        (VarType::U64, VarType::F64) => {
                            writeln!(s, "\t{glsl_ty} {dst} = doubleBitsToUint64({src});")
                        }
                        (
                            VarType::I8
                            | VarType::U8
                            | VarType::I16
                            | VarType::U16
                            | VarType::I32
                            | VarType::U32
                            | VarType::I64
                            | VarType::U64,
                            VarType::I8
                            | VarType::U8
                            | VarType::I16
                            | VarType::U16
                            | VarType::I32
                            | VarType::U32
                            | VarType::I64
                            | VarType::U64,
                        ) => writeln!(s, "\t{glsl_ty} {dst} = {glsl_ty}({src});"),
                        _ => todo!(),
                    },
                    crate::op::Uop::Neg => match ty {
                        VarType::Bool => writeln!(s, "\t{glsl_ty} {dst} = !{src};"),
                        _ => writeln!(s, "\t{glsl_ty} {dst} = -{src};"),
                    },
                    crate::op::Uop::Sqrt => writeln!(s, "\t{glsl_ty} {dst} = sqrt({src});"),
                    crate::op::Uop::Abs => writeln!(s, "\t{glsl_ty} {dst} = abs({src});"),
                    crate::op::Uop::Sin => writeln!(s, "\t{glsl_ty} {dst} = sin({src});"),
                    crate::op::Uop::Cos => writeln!(s, "\t{glsl_ty} {dst} = cos({src});"),
                    crate::op::Uop::Exp2 => writeln!(s, "\t{glsl_ty} {dst} = exp2({src});"),
                    crate::op::Uop::Log2 => writeln!(s, "\t{glsl_ty} {dst} = log2({src});"),
                }?;
            }
            crate::op::KernelOp::FMA => {}
            _ => {}
        }
    }
    Ok(())
}
fn assemble_cast_inline(
    s: &mut String,
    src: impl Display,
    dst_ty: &'static VarType,
    src_ty: &VarType,
) -> std::fmt::Result {
    match (dst_ty, src_ty) {
        (dst_ty, src_ty) if dst_ty == src_ty => {
            // Trivial Cast
            write!(s, "{src}")
        }
        (
            VarType::Bool
            | VarType::U8
            | VarType::I8
            | VarType::U16
            | VarType::I16
            | VarType::U32
            | VarType::I32
            | VarType::U64
            | VarType::I64
            | VarType::F16
            | VarType::F32
            | VarType::F64,
            VarType::Bool
            | VarType::U8
            | VarType::I8
            | VarType::U16
            | VarType::I16
            | VarType::U32
            | VarType::I32
            | VarType::U64
            | VarType::I64
            | VarType::F16
            | VarType::F32
            | VarType::F64,
        ) => {
            // Primary Type Casting
            write!(s, "{dst_ty}({src})", dst_ty = GlslTypeName(dst_ty))
        }
        (
            VarType::Vec {
                ty: vec_ty,
                num: vec_num,
            },
            VarType::Array {
                ty: arr_ty,
                num: arr_num,
            },
        ) => {
            write!(s, "{dst_ty}(", dst_ty = GlslTypeName(dst_ty))?;
            assert_eq!(vec_num, arr_num);
            assert!(*vec_num <= 4);
            // let swizzles = ["x", "y", "z", "w"];
            for i in 0..*vec_num {
                write!(s, "{src}[{i}]")?;
                if i < vec_num - 1 {
                    write!(s, ",")?;
                }
            }
            write!(s, ")")?;
            Ok(())
        }
        (
            VarType::Array {
                ty: arr_ty,
                num: arr_num,
            },
            VarType::Vec {
                ty: vec_ty,
                num: vec_num,
            },
        ) => {
            write!(s, "{dst_ty}(", dst_ty = GlslTypeName(dst_ty))?;
            assert_eq!(vec_num, arr_num);
            assert!(*vec_num <= 4);
            let swizzles = ["x", "y", "z", "w"];
            for i in 0..*vec_num {
                assemble_cast_inline(
                    s,
                    format!("{src}.{swizzle}", swizzle = swizzles[i]),
                    arr_ty,
                    vec_ty,
                )?;
                if i < vec_num - 1 {
                    write!(s, ",")?;
                }
            }
            write!(s, ")")?;
            Ok(())
        }
        (VarType::Struct{tys: dst_tys}, VarType::Struct{tys:src_tys})=>{
            write!(s, "{dst_ty}(", dst_ty = GlslTypeName(dst_ty))?;
            for (i, (dst_ty, src_ty)) in dst_tys.into_iter().zip(src_tys.into_iter()).enumerate(){
                assemble_cast_inline(s, format!("{src}.e{i}"), dst_ty, src_ty)?;
                if i < dst_tys.len()-1{
                    write!(s, ", ")?;
                }
            }
            write!(s, ")")?;
            Ok(())
        }
        _ => todo!("Cast between {src_ty:?} -> {dst_ty:?} has not been implemented for the GLSL codegen backend."),
    }
}
fn assemble_cast(
    s: &mut String,
    dst: Reg,
    src: Reg,
    dst_ty: &'static VarType,
    src_ty: &VarType,
) -> std::fmt::Result {
    write!(s, "\t{dst_ty} {dst} = ", dst_ty = GlslTypeName(dst_ty))?;
    assemble_cast_inline(s, src, dst_ty, src_ty)?;
    writeln!(s, ";")?;
    Ok(())
}

pub fn zeroable(ty: &VarType) -> bool {
    matches!(
        ty,
        VarType::Bool
            | VarType::I8
            | VarType::U8
            | VarType::I16
            | VarType::U16
            | VarType::I32
            | VarType::U32
            | VarType::I64
            | VarType::U64
            | VarType::F16
            | VarType::F32
            | VarType::F64
            | VarType::Vec { .. }
            | VarType::Mat { .. }
    )
}
