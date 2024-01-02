mod glslext;

use super::CompileInfo;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::ir::{VarId, IR};
use crate::op::{Bop, KernelOp, ReduceOp, Uop};
use crate::vartype::{self, AsVarType, Intersection, VarType};
use glslext::GLSL450Instruction;
use rspirv::binary::{Assemble, Disassemble};
use rspirv::{dr, spirv};

pub const BUFFER_BINDING: u32 = 0;
pub const TEXTURE_BINDING: u32 = 1;
pub const ACCEL_BINDING: u32 = 2;

// fn ty(ty: &VarType) ->

// #[derive(Debug, thiserror::Error)]
// pub enum Error {
//     #[error("{0}")]
//     RspirvError(#[from] dr::Error, Backtrace),
// }
//

macro_rules! glsl_ext {
    ($self:ident, $ty:ident, $ext:ident, $($operand:expr),*) => {
        {
        let ext = $self.glsl_ext;
        $self.ext_inst(
            $ty,
            None,
            ext,
            GLSL450Instruction::$ext as _,
            [$(dr::Operand::IdRef($operand)),*],
        )
        }
    };
}

pub fn assemble_trace(
    trace: &IR,
    info: &CompileInfo,
    entry_point: &str,
) -> Result<Vec<u32>, dr::Error> {
    let mut b = SpirvBuilder::default();
    b.assemble(trace, info, entry_point)?;

    let module = b.module();
    log::trace!("Shader: \n {}", module.disassemble());

    let spv = module.assemble();

    use spirv_tools::val::compiled::CompiledValidator;
    use spirv_tools::val::{Validator, ValidatorOptions};

    let val = CompiledValidator::default();
    val.validate(
        &spv,
        Some(ValidatorOptions {
            ..Default::default()
        }),
    )
    .map_err(|err| {
        log::error!("Spirv Validation {err}");
    })
    .ok();

    Ok(spv)
}
fn isfloat(ty: &VarType) -> bool {
    match ty {
        VarType::F16 | VarType::F32 | VarType::F64 => true,
        _ => false,
    }
}
fn isint(ty: &VarType) -> bool {
    match ty {
        VarType::I8
        | VarType::U8
        | VarType::I16
        | VarType::U16
        | VarType::I32
        | VarType::U32
        | VarType::I64
        | VarType::U64 => true,
        _ => false,
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SamplerDesc {
    ty: &'static VarType,
    dim: spirv::Dim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Requirements {
    int_8: bool,
    int_16: bool,
    int_64: bool,
    float_16: bool,
    float_64: bool,
    atomic_float_16_add: bool,
    atomic_float_32_add: bool,
    atomic_float_64_add: bool,
    atomic_float_16_min_max: bool,
    atomic_float_32_min_max: bool,
    atomic_float_64_min_max: bool,
    ray_query: bool,
}

/// A wrapper arround `dr::Builder` to generate spriv code.
///
/// * `b`: Builder from `rspriv`
/// * `spirv_vars`: Mapping from VarId -> Spirv Id
/// * `glsl_ext`: GLSL_EXT libarary loaded by default
#[derive(Default)]
struct SpirvBuilder {
    b: dr::Builder,
    spriv_regs: HashMap<VarId, u32>,

    // Maps spirv type to a arry of buffers with that type
    buffer_arrays: HashMap<u32, u32>,
    accel_var: Option<u32>,
    samplers: HashMap<SamplerDesc, u32>,

    glsl_ext: u32,
    function_block: usize,
    interface_vars: Vec<u32>,

    types: HashMap<&'static VarType, u32>,

    n_buffers: usize,
    n_textures: usize,
    n_accels: usize,
}
impl Deref for SpirvBuilder {
    type Target = dr::Builder;

    fn deref(&self) -> &Self::Target {
        &self.b
    }
}
impl DerefMut for SpirvBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.b
    }
}

impl SpirvBuilder {
    pub fn module(self) -> dr::Module {
        self.b.module()
    }
    pub fn assemble(
        &mut self,
        ir: &IR,
        info: &CompileInfo,
        entry_point: &str,
    ) -> Result<(), dr::Error> {
        self.n_buffers = 1 + ir.n_buffers; // [size_buffer, ir buffers]
        self.n_textures = ir.n_textures;
        self.n_accels = ir.n_accels;

        // Advance Id idk. why this didn't work without
        self.id();

        self.set_version(1, 5);

        // Enable capabilities
        self.capability(spirv::Capability::Shader);
        self.capability(spirv::Capability::StorageUniformBufferBlock16);
        self.capability(spirv::Capability::Int8);
        self.capability(spirv::Capability::Int16);
        self.capability(spirv::Capability::Int64);
        self.capability(spirv::Capability::Float16);
        self.capability(spirv::Capability::Float64);
        self.capability(spirv::Capability::AtomicFloat16AddEXT);
        self.capability(spirv::Capability::AtomicFloat32AddEXT);
        self.capability(spirv::Capability::AtomicFloat64AddEXT);
        self.capability(spirv::Capability::AtomicFloat16MinMaxEXT);
        self.capability(spirv::Capability::AtomicFloat32MinMaxEXT);
        self.capability(spirv::Capability::AtomicFloat64MinMaxEXT);
        self.capability(spirv::Capability::Int64Atomics);

        // Add ray query capability only if it is needed
        // TODO: Refactor into properties?
        if ir.vars.iter().any(|var| var.op == KernelOp::TraceRay) {
            self.capability(spirv::Capability::RayQueryKHR);
        }

        // Enable Extensions
        self.glsl_ext = self.ext_inst_import("GLSL.std.450");

        self.extension("SPV_KHR_16bit_storage");
        self.extension("SPV_KHR_ray_query");
        self.extension("SPV_EXT_shader_atomic_float_add");
        self.extension("SPV_EXT_shader_atomic_float16_add");
        self.extension("SPV_EXT_shader_atomic_float_min_max");

        self.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        let void = self.type_void();
        let voidf = self.type_function(void, vec![]);

        let u32_ty = self.type_int(32, 0);
        let v3u32_ty = self.type_vector(u32_ty, 3);
        let ptr_v3u32_ty = self.type_pointer(None, spirv::StorageClass::Input, v3u32_ty);
        let global_invocation_id =
            self.variable(ptr_v3u32_ty, None, spirv::StorageClass::Input, None);

        self.decorate(
            global_invocation_id,
            spirv::Decoration::BuiltIn,
            [dr::Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)],
        );

        // let storage_vars = self.assemble_storage_vars(ir);

        let func = self.begin_function(
            void,
            None,
            spirv::FunctionControl::DONT_INLINE | spirv::FunctionControl::CONST,
            voidf,
        )?;

        self.begin_block(None)?;
        self.function_block = self.selected_block().unwrap();

        // Load Kernel size from size_buffer
        let buffer_idx = 0;
        let u32_ty = self.type_int(32, 0);
        let ptr_ty = self.type_pointer(None, spirv::StorageClass::StorageBuffer, u32_ty);
        let buffer_idx = self.constant_bit32(u32_ty, buffer_idx);
        let elem = self.constant_bit32(u32_ty, 0);
        let buffer_array = self.get_buffer_array(&VarType::U32);
        let idx = self.constant_bit32(u32_ty, 0);
        let ptr = self.access_chain(ptr_ty, None, buffer_array, [buffer_idx, elem, idx])?;
        let size = self.load(u32_ty, None, ptr, None, None)?;

        // Load index
        // TODO: refactor into function
        let u32_ty = self.type_int(32, 0);
        let ptr_ty = self.type_pointer(None, spirv::StorageClass::Input, u32_ty);
        let u32_0 = self.constant_bit32(u32_ty, 0);
        let ptr = self.access_chain(ptr_ty, None, global_invocation_id, [u32_0])?;
        let index = self.load(u32_ty, None, ptr, None, None)?;

        let bool_ty = self.type_bool();
        let cond = self.u_greater_than_equal(bool_ty, None, index, size)?;

        // Early exit if index >= size
        self.return_if(cond)?;

        self.assemble_vars(ir, global_invocation_id)?;

        self.ret()?;
        self.end_function()?;

        let interface = [global_invocation_id]
            .into_iter()
            .chain(self.interface_vars.drain(..))
            .collect::<Vec<_>>();

        self.entry_point(
            spirv::ExecutionModel::GLCompute,
            func,
            entry_point,
            interface,
        );
        //OpEntryPoint GLCompute %main "main" %test %gl_GlobalInvocationID %test2
        // TODO: get prefered local size from device (somehow)
        self.execution_mode(
            func,
            spirv::ExecutionMode::LocalSize,
            [info.work_group_size, 1, 1],
        );

        Ok(())
    }
    fn reg(&self, id: VarId) -> u32 {
        self.spriv_regs[&id]
    }
    pub fn buffer_idx(&self, data_idx: u64) -> u32 {
        // Offset buffer index by 1 for the size buffer
        (data_idx + 1) as _
    }
    /// Put the instructions at the beginning of a function
    fn with_function<T, F: FnOnce(&mut Self) -> Result<T, dr::Error>>(
        &mut self,
        f: F,
    ) -> Result<T, dr::Error> {
        let current = self.selected_block();
        let function_block = self.function_block;
        self.select_block(Some(function_block))?;
        let res = f(self)?;
        self.select_block(current)?;
        Ok(res)
    }
    /// Put the instructions at the top of the module, outside of any function
    fn with_module<T, F: FnOnce(&mut Self) -> Result<T, dr::Error>>(
        &mut self,
        f: F,
    ) -> Result<T, dr::Error> {
        let current = self.selected_block();
        self.select_block(None)?;
        let res = f(self)?;
        self.select_block(current)?;
        Ok(res)
    }

    pub fn selection_merge(
        &mut self,
        merge_block: spirv::Word,
        selection_control: spirv::SelectionControl,
    ) -> Result<(), dr::Error> {
        // According to spirv OpSelectionMerge should be second to last
        // instruction in block. Rspirv however ends block with
        // selection_merge. Therefore, we insert the instruction by hand.
        self.b.insert_into_block(
            rspirv::dr::InsertPoint::End,
            rspirv::dr::Instruction::new(
                spirv::Op::SelectionMerge,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(merge_block),
                    rspirv::dr::Operand::SelectionControl(selection_control),
                ],
            ),
        )
    }
    fn while_block(
        &mut self,
        cond: impl FnOnce(&mut Self) -> Result<u32, dr::Error>,
        f: impl FnOnce(&mut Self) -> Result<(), dr::Error>,
    ) -> Result<(), dr::Error> {
        let start_label = self.id();
        let start_cond_label = self.id();
        let start_body_label = self.id();
        let continue_label = self.id();
        let end_label = self.id();

        self.branch(start_label)?;
        self.begin_block(Some(start_label))?;

        // According to spirv OpLoopMerge should be second to last
        // instruction in block. Rspirv however ends block with
        // selection_merge. Therefore, we insert the instruction by hand.
        self.b
            .insert_into_block(
                rspirv::dr::InsertPoint::End,
                rspirv::dr::Instruction::new(
                    spirv::Op::LoopMerge,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(end_label),
                        rspirv::dr::Operand::IdRef(continue_label),
                        rspirv::dr::Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
            )
            .unwrap();

        self.branch(start_cond_label)?;
        self.begin_block(Some(start_cond_label))?;

        let cond = cond(self)?;

        self.branch_conditional(cond, start_body_label, end_label, [])?;
        self.begin_block(Some(start_body_label))?;

        f(self)?;

        self.branch(continue_label)?;
        self.begin_block(Some(continue_label))?;

        self.branch(start_label)?;

        self.begin_block(Some(end_label))?;

        Ok(())
    }
    fn return_if(&mut self, cond: u32) -> Result<u32, dr::Error> {
        let true_label = self.id();
        let false_label = self.id();

        self.selection_merge(false_label, spirv::SelectionControl::NONE)?;
        self.branch_conditional(cond, true_label, false_label, [])?;

        self.begin_block(Some(true_label))?;

        self.ret()?;

        self.begin_block(Some(false_label))?;

        Ok(true_label)
    }
    /// Represents an if condition.
    /// Returns the label of the true bloc
    fn if_block(&mut self, cond: u32, f: impl FnOnce(&mut Self)) -> Result<u32, dr::Error> {
        let true_label = self.id();
        let false_label = self.id();

        self.selection_merge(false_label, spirv::SelectionControl::NONE)?;
        self.branch_conditional(cond, true_label, false_label, [])?;

        self.begin_block(Some(true_label))?;

        f(self);

        self.branch(false_label)?;

        self.begin_block(Some(false_label))?;

        Ok(true_label)
    }
    /// Store value at src in ptr, potentially atomically
    fn store_reduce(
        &mut self,
        src: spirv::Word,
        ptr: spirv::Word,
        ty: &'static VarType,
        op: &Option<ReduceOp>,
    ) -> Result<Option<u32>, dr::Error> {
        let spirv_ty = self.spirv_ty(ty);
        let u32_ty = self.type_int(32, 0);
        let u32_0 = self.constant_bit32(u32_ty, 0);
        let u32_1 = self.constant_bit32(u32_ty, 1);
        match ty {
            VarType::Bool => {
                let u8_ty = self.type_int(8, 0);
                let u8_0 = self.constant_bit32(u8_ty, 0);
                let u8_1 = self.constant_bit32(u8_ty, 1);

                let data = self.select(u8_ty, None, src, u8_1, u8_0)?;
                self.store(ptr, data, None, None)?;
                Ok(None)
            }
            _ => match op {
                Some(op) => {
                    let res =
                            match op {
                                ReduceOp::Sum => match ty {
                                    VarType::I8
                                    | VarType::U8
                                    | VarType::I16
                                    | VarType::U16
                                    | VarType::I32
                                    | VarType::U32
                                    | VarType::I64
                                    | VarType::U64 => {
                                        self.atomic_i_add(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    VarType::F16| VarType::F32 | VarType::F64 => self
                                        .atomic_f_add_ext(spirv_ty, None, ptr, u32_1, u32_0, src)?,
                                    _ => todo!(),
                                },
                                ReduceOp::And => match ty {
                                    VarType::U8
                                    | VarType::U16
                                    | VarType::U32
                                    | VarType::U64 => {
                                        self.atomic_and(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    _ => todo!(),
                                },
                                ReduceOp::Or => match ty {
                                    VarType::U8
                                    | VarType::U16
                                    | VarType::U32
                                    | VarType::U64 => {
                                        self.atomic_or(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    _ => todo!(),
                                },
                                ReduceOp::Xor => match ty {
                                    VarType::U8
                                    | VarType::U16
                                    | VarType::U32
                                    | VarType::U64 => {
                                        self.atomic_xor(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    _ => todo!(),
                                },
                                ReduceOp::Min => match ty {
                                    VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                        self.atomic_u_min(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                        self.atomic_s_min(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    VarType::F16|VarType::F32 | VarType::F64 => self
                                        .atomic_f_min_ext(spirv_ty, None, ptr, u32_1, u32_0, src)?,
                                    _ => todo!(),
                                },
                                ReduceOp::Max => match ty {
                                    VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                        self.atomic_u_max(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                        self.atomic_s_max(spirv_ty, None, ptr, u32_1, u32_0, src)?
                                    }
                                    VarType::F16|VarType::F32 | VarType::F64 => self
                                        .atomic_f_max_ext(spirv_ty, None, ptr, u32_1, u32_0, src)?,
                                    _ => todo!(),
                                },
                                _ => todo!("{op:?} is not supported as an atomic scatter operation by the spirv backend!"),
                            };
                    Ok(Some(res))
                }
                None => {
                    self.store(ptr, src, None, None)?;
                    Ok(None)
                }
            },
        }
    }
    fn spirv_ty(&mut self, ty: &'static VarType) -> u32 {
        // Deduplicate types
        if let Some(ty) = self.types.get(ty) {
            *ty
        } else {
            let spirv_ty = match ty {
                VarType::Void => self.type_void(),
                VarType::Bool => self.type_bool(),
                VarType::I8 => self.type_int(8, 1),
                VarType::U8 => self.type_int(8, 0),
                VarType::I16 => self.type_int(16, 1),
                VarType::U16 => self.type_int(16, 0),
                VarType::I32 => self.type_int(32, 1),
                VarType::U32 => self.type_int(32, 0),
                VarType::I64 => self.type_int(64, 1),
                VarType::U64 => self.type_int(64, 0),
                VarType::F16 => self.type_float(16),
                VarType::F32 => self.type_float(32),
                VarType::F64 => self.type_float(64),
                VarType::Vec { ty, num } => {
                    let ty = self.spirv_ty(ty);
                    self.type_vector(ty, *num as _)
                }
                VarType::Struct { tys } => {
                    let spv_tys = tys.iter().map(|ty| self.spirv_ty(ty)).collect::<Vec<_>>();
                    let struct_ty = self.type_struct(spv_tys);
                    for i in 0..tys.len() {
                        let offset = ty.offset(i);
                        self.member_decorate(
                            struct_ty,
                            i as _,
                            spirv::Decoration::Offset,
                            [dr::Operand::LiteralBit32(offset as _)],
                        );
                        match tys[i] {
                            VarType::Mat { .. } => self.member_decorate(
                                struct_ty,
                                i as _,
                                spirv::Decoration::RowMajor,
                                [],
                            ),
                            _ => {}
                        }
                    }
                    struct_ty
                }
                VarType::Mat { ty, cols, rows } => {
                    let vec_ty = self.spirv_ty(vartype::vector(ty, *rows));
                    self.type_matrix(vec_ty, *cols as _)
                }
                VarType::Array { ty, num } => {
                    let ty = self.spirv_ty(ty);
                    let u32_ty = self.type_int(32, 0);
                    let num = self.constant_bit32(u32_ty, *num as _);
                    self.type_array(ty, num)
                }
            };
            self.types.insert(ty, spirv_ty);
            spirv_ty
        }
    }
    fn get_samplers(&mut self, desc: SamplerDesc) -> u32 {
        if let Some(sampler) = self.samplers.get(&desc) {
            *sampler
        } else {
            let n_textures = self.n_textures;
            let ty = self.spirv_ty(&desc.ty);
            let ty_image =
                self.type_image(ty, desc.dim, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
            let ty_sampled_image = self.type_sampled_image(ty_image);

            let ty_int = self.type_int(32, 0);
            let length = self.constant_bit32(ty_int, n_textures as _);

            let ty_array = self.type_array(ty_sampled_image, length);
            let ty_ptr = self.type_pointer(None, spirv::StorageClass::UniformConstant, ty_array);

            let res = self
                .with_module(|s| {
                    let res = s.variable(ty_ptr, None, spirv::StorageClass::UniformConstant, None);

                    s.decorate(
                        res,
                        spirv::Decoration::DescriptorSet,
                        [dr::Operand::LiteralBit32(0)],
                    );
                    s.decorate(
                        res,
                        spirv::Decoration::Binding,
                        [dr::Operand::LiteralBit32(1)],
                    );
                    Ok(res)
                })
                .unwrap();

            self.samplers.insert(desc.clone(), res);
            self.interface_vars.push(res);
            res
        }
    }
    /// Adds the AccelerationStructure array variable to the builder
    /// TODO: rename
    fn get_accel_array(&mut self) -> u32 {
        if let Some(res) = &self.accel_var {
            *res
        } else {
            let n_accels = self.n_accels;
            let accel_ty = self.type_acceleration_structure_khr();
            let u32_ty = self.type_int(32, 0);
            let array_len = self.constant_bit32(u32_ty, n_accels as _);
            let array_ty = self.type_array(accel_ty, array_len);
            let ptr_ty = self.type_pointer(None, spirv::StorageClass::UniformConstant, array_ty);
            let res = self
                .with_module(|s| {
                    let res = s.variable(ptr_ty, None, spirv::StorageClass::UniformConstant, None);

                    s.decorate(
                        res,
                        spirv::Decoration::DescriptorSet,
                        [dr::Operand::LiteralBit32(0)],
                    );
                    s.decorate(
                        res,
                        spirv::Decoration::Binding,
                        [dr::Operand::LiteralBit32(ACCEL_BINDING)],
                    );
                    s.interface_vars.push(res);
                    Ok(res)
                })
                .unwrap();
            self.accel_var = Some(res);
            res
        }
    }
    /// Returns the variable, representing the binding to which the buffers are bound of the type
    /// `ty`.
    /// It only inserts the code if the variable does not exist for that type.
    ///
    ///
    /// * `ty`: Type for which to generate the buffer array
    /// TODO: rename
    fn get_buffer_array(&mut self, ty: &'static VarType) -> u32 {
        let ty = match ty {
            VarType::Bool => u8::var_ty(),
            _ => ty,
        };
        let spirv_ty = self.spirv_ty(&ty);
        let n_buffers = self.n_buffers;

        if self.buffer_arrays.contains_key(&spirv_ty) {
            self.buffer_arrays[&spirv_ty]
        } else {
            let u32_ty = self.type_int(32, 0);
            let array_len = self.constant_bit32(u32_ty, n_buffers as _);
            let rta_ty = self.type_runtime_array(spirv_ty);
            let struct_ty = self.type_struct([rta_ty]);
            let array_ty = self.type_array(struct_ty, array_len);

            self.decorate(struct_ty, spirv::Decoration::Block, []);
            self.member_decorate(
                struct_ty,
                0,
                spirv::Decoration::Offset,
                [dr::Operand::LiteralBit32(0)],
            );
            self.decorate(
                rta_ty,
                rspirv::spirv::Decoration::ArrayStride,
                [dr::Operand::LiteralBit32(ty.size() as _)],
            );

            let ptr_ty = self.type_pointer(None, spirv::StorageClass::StorageBuffer, array_ty);

            let res = self
                .with_module(|s| {
                    let res = s.variable(ptr_ty, None, spirv::StorageClass::StorageBuffer, None);

                    s.decorate(
                        res,
                        spirv::Decoration::DescriptorSet,
                        [dr::Operand::LiteralBit32(0)],
                    );
                    s.decorate(
                        res,
                        spirv::Decoration::Binding,
                        [dr::Operand::LiteralBit32(BUFFER_BINDING)],
                    );
                    Ok(res)
                })
                .unwrap();

            self.buffer_arrays.insert(spirv_ty, res);
            self.interface_vars.push(res);
            res
        }
    }
    fn composite_construct(
        &mut self,
        ty: &'static VarType,
        elems: impl IntoIterator<Item = u32>,
    ) -> Result<u32, dr::Error> {
        // let elems = elems.into_iter();
        let ty = self.spirv_ty(&ty);
        // let deps = deps.iter().map(|id| self.reg(*id)).collect::<Vec<_>>();
        self.b.composite_construct(ty, None, elems)
    }

    fn assemble_vars(&mut self, ir: &IR, global_invocation_id: u32) -> Result<(), dr::Error> {
        for varid in ir.var_ids() {
            let var = ir.var(varid);
            let deps = ir.deps(varid);
            let res = match var.op {
                KernelOp::Nop => 0,
                KernelOp::Bop(op) => {
                    let src_ty = &ir.var(deps[0]).ty;
                    let lhs = self.reg(deps[0]);
                    let rhs = self.reg(deps[1]);

                    self.bop(op, var.ty, &src_ty, lhs, rhs)?
                }
                KernelOp::Uop(op) => {
                    // let res = self.id();
                    let src = self.reg(deps[0]);
                    let ty = &var.ty;
                    let src_ty = &ir.var(deps[0]).ty;
                    let spirv_ty = self.spirv_ty(&var.ty);

                    match op {
                        Uop::Cast => match (src_ty, ty) {
                            (
                                VarType::F16 | VarType::F32 | VarType::F64,
                                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
                            ) => self.convert_f_to_u(spirv_ty, None, src)?,
                            (
                                VarType::F16 | VarType::F32 | VarType::F64,
                                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
                            ) => self.convert_f_to_s(spirv_ty, None, src)?,
                            (
                                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
                                VarType::F16 | VarType::F32 | VarType::F64,
                            ) => self.convert_u_to_f(spirv_ty, None, src)?,
                            (
                                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
                                VarType::F16 | VarType::F32 | VarType::F64,
                            ) => self.convert_s_to_f(spirv_ty, None, src)?,
                            (
                                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
                                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
                            ) => self.s_convert(spirv_ty, None, src)?,
                            (
                                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
                                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
                            ) => self.u_convert(spirv_ty, None, src)?,
                            (
                                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
                                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
                            ) => {
                                let s = self.s_convert(spirv_ty, None, src)?;
                                todo!("Implemente S->U cast")
                            }
                            (
                                VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64,
                                VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64,
                            ) => {
                                let u = self.u_convert(spirv_ty, None, src)?;
                                todo!("Implemente U->S cast")
                            }
                            _ => todo!(),
                        },
                        Uop::BitCast => self.bitcast(spirv_ty, None, src)?,
                        Uop::Neg => match ty {
                            VarType::Bool => self.logical_not(spirv_ty, None, src)?,
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                self.s_negate(spirv_ty, None, src)?
                            }
                            VarType::F16 | VarType::F32 | VarType::F64 => {
                                self.f_negate(spirv_ty, None, src)?
                            }
                            _ => todo!(),
                        },
                        Uop::Sqrt => glsl_ext!(self, spirv_ty, Sqrt, src)?,
                        Uop::Abs => match ty {
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                glsl_ext!(self, spirv_ty, SAbs, src)?
                            }
                            VarType::F16 | VarType::F32 | VarType::F64 => {
                                glsl_ext!(self, spirv_ty, FAbs, src)?
                            }
                            _ => todo!(),
                        },
                        Uop::Sin => match ty {
                            VarType::F16 | VarType::F32 | VarType::F64 => {
                                glsl_ext!(self, spirv_ty, Sin, src)?
                            }
                            _ => todo!(),
                        },
                        Uop::Cos => match ty {
                            VarType::F16 | VarType::F32 | VarType::F64 => {
                                glsl_ext!(self, spirv_ty, Cos, src)?
                            }
                            _ => todo!(),
                        },
                        Uop::Exp2 => match ty {
                            VarType::F16 | VarType::F32 | VarType::F64 => {
                                glsl_ext!(self, spirv_ty, Exp2, src)?
                            }
                            _ => todo!(),
                        },
                        Uop::Log2 => match ty {
                            VarType::F16 | VarType::F32 | VarType::F64 => {
                                glsl_ext!(self, spirv_ty, Log2, src)?
                            }
                            _ => todo!(),
                        },
                    }
                }
                KernelOp::Select => {
                    let ty = self.spirv_ty(&var.ty);
                    let cond = self.reg(deps[0]);
                    let true_val = self.reg(deps[1]);
                    let false_val = self.reg(deps[2]);
                    self.select(ty, None, cond, true_val, false_val)?
                }
                KernelOp::Construct => {
                    let deps = deps.iter().map(|id| self.reg(*id)).collect::<Vec<_>>();
                    self.composite_construct(&var.ty, deps)?
                }
                KernelOp::Extract(elem) => {
                    let ty = self.spirv_ty(&ir.var(varid).ty);
                    let src = self.reg(deps[0]);
                    self.composite_extract(ty, None, src, [elem as u32])?
                }
                KernelOp::TraceRay => {
                    let accels = deps[0];
                    let accel_idx = ir.var(accels).data;
                    let o = deps[1];
                    let d = deps[2];
                    let tmin = deps[3];
                    let tmax = deps[4];

                    let u32_ty = self.type_int(32, 0);
                    let i32_ty = self.type_int(32, 1);

                    let accels = self.get_accel_array();
                    let accel_idx = self.constant_bit32(u32_ty, accel_idx as _);
                    let accel_ty = self.type_acceleration_structure_khr();
                    let accel_ptr_ty =
                        self.type_pointer(None, spirv::StorageClass::UniformConstant, accel_ty);
                    let accel_ptr = self.access_chain(accel_ptr_ty, None, accels, [accel_idx])?;

                    let accel = self.load(accel_ty, None, accel_ptr, None, None)?;

                    let ray_query_ty = self.type_ray_query_khr();
                    let ray_query_ptr_ty =
                        self.type_pointer(None, spirv::StorageClass::Private, ray_query_ty);

                    let ray_query_var = self.with_module(|s| {
                        let var =
                            s.variable(ray_query_ptr_ty, None, spirv::StorageClass::Private, None);
                        s.interface_vars.push(var);
                        Ok(var)
                    })?;

                    let ray_flags = self.constant_bit32(u32_ty, 0x01);
                    let cull_mask = self.constant_bit32(u32_ty, 0xff);
                    let o = self.reg(o);
                    let d = self.reg(d);
                    let tmin = self.reg(tmin);
                    let tmax = self.reg(tmax);

                    self.ray_query_initialize_khr(
                        ray_query_var,
                        accel,
                        ray_flags,
                        cull_mask,
                        o,
                        tmin,
                        d,
                        tmax,
                    )?;

                    let bool_ty = self.type_bool();
                    let i32_0 = self.constant_bit32(i32_ty, 0);
                    let u32_0 = self.constant_bit32(u32_ty, 0);

                    // TODO: Custom intersections (may require ray-tracing shader)
                    self.while_block(
                        |s| s.ray_query_proceed_khr(bool_ty, None, ray_query_var),
                        |s| {
                            let intersection_ty = s.ray_query_get_intersection_type_khr(
                                u32_ty,
                                None,
                                ray_query_var,
                                i32_0,
                            )?;
                            let is_opaque = s.i_equal(bool_ty, None, intersection_ty, u32_0)?;
                            s.if_block(is_opaque, |s| {
                                s.ray_query_confirm_intersection_khr(ray_query_var).unwrap();
                            });
                            Ok(())
                        },
                    )?;

                    let vec2_ty = self.spirv_ty(vartype::vector(f32::var_ty(), 2));

                    let i32_1 = self.constant_bit32(i32_ty, 1);

                    let instance_id = self.ray_query_get_intersection_instance_id_khr(
                        u32_ty,
                        None,
                        ray_query_var,
                        i32_1,
                    )?;
                    let primitive_idx = self.ray_query_get_intersection_primitive_index_khr(
                        u32_ty,
                        None,
                        ray_query_var,
                        i32_1,
                    )?;
                    let barycentrics = self.ray_query_get_intersection_barycentrics_khr(
                        vec2_ty,
                        None,
                        ray_query_var,
                        i32_1,
                    )?;
                    let intersection_type_khr = self.ray_query_get_intersection_type_khr(
                        u32_ty,
                        None,
                        ray_query_var,
                        i32_1,
                    )?;

                    let intersection_ty = Intersection::var_ty();

                    let intersection = self.composite_construct(
                        intersection_ty,
                        [
                            barycentrics,
                            instance_id,
                            primitive_idx,
                            intersection_type_khr,
                        ],
                    )?;

                    intersection
                }
                KernelOp::TexLookup => {
                    let img = deps[0];
                    let coord = self.reg(deps[1]);

                    let dim = match ir.var(img).op {
                        KernelOp::TextureRef { dim } => match dim {
                            1 => spirv::Dim::Dim1D,
                            2 => spirv::Dim::Dim2D,
                            3 => spirv::Dim::Dim3D,
                            _ => todo!(),
                        },
                        _ => todo!(),
                    };

                    let img_idx = ir.var(img).data;

                    let ty = self.spirv_ty(ir.var_ty(img));
                    let ty_image =
                        self.type_image(ty, dim, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
                    let ty_sampled_image = self.type_sampled_image(ty_image);
                    let ptr_ty = self.type_pointer(
                        None,
                        spirv::StorageClass::UniformConstant,
                        ty_sampled_image,
                    );

                    let int_ty = self.type_int(32, 0);
                    let img_idx = self.constant_bit32(int_ty, img_idx as _);

                    let samplers = self.get_samplers(SamplerDesc {
                        ty: ir.var_ty(img),
                        dim,
                    });
                    let ptr = self.access_chain(ptr_ty, None, samplers, [img_idx])?;

                    let ty_v4 = self.type_vector(ty, 4);

                    let float_ty = self.type_float(32);
                    let float_0 = self.constant_bit32(float_ty, 0f32.to_bits());

                    let img = self.load(ty_sampled_image, None, ptr, None, None)?;

                    self.image_sample_explicit_lod(
                        ty_v4,
                        // None,
                        None,
                        img,
                        coord,
                        spirv::ImageOperands::LOD,
                        [dr::Operand::IdRef(float_0)],
                    )?
                }
                KernelOp::Scatter(reduce_op) => {
                    let dst = deps[0];
                    let src = deps[1];
                    let idx = deps[2];
                    let cond = deps.get(3);
                    let ty = &ir.var(src).ty;
                    let data_ty = match ty {
                        VarType::Bool => &VarType::U8,
                        _ => &ir.var(src).ty,
                    };

                    let buffer_idx = self.buffer_idx(ir.var(dst).data);

                    let spriv_ty = self.spirv_ty(data_ty);
                    let ptr_ty =
                        self.type_pointer(None, spirv::StorageClass::StorageBuffer, spriv_ty);
                    let int_ty = self.type_int(32, 0);
                    let buffer = self.constant_bit32(int_ty, buffer_idx);
                    let elem = self.constant_bit32(int_ty, 0);

                    let buffer_array = self.get_buffer_array(&ty);
                    let idx = self.reg(idx);
                    let src = self.reg(src);

                    // Don't need to condition the scatter if that's not neccesarry
                    // TODO: unify conditioned and uncoditioned part
                    if let Some(cond) = cond {
                        assert_eq!(ir.var(*cond).ty, bool::var_ty());

                        let cond = self.reg(*cond);
                        self.if_block(cond, |s| {
                            let ptr = s
                                .access_chain(ptr_ty, None, buffer_array, [buffer, elem, idx])
                                .unwrap();

                            // TODO: suport returning atomic in if block
                            s.store_reduce(src, ptr, ty, &reduce_op)
                                .unwrap()
                                .unwrap_or(0);
                        });
                        0
                    } else {
                        let ptr =
                            self.access_chain(ptr_ty, None, buffer_array, [buffer, elem, idx])?;

                        self.store_reduce(src, ptr, ty, &reduce_op)?.unwrap_or(0)
                    }
                }
                KernelOp::Gather => {
                    let src = deps[0];
                    let idx = deps[1];
                    let cond = deps.get(2);
                    let buffer_idx = self.buffer_idx(ir.var(src).data);

                    let ty = &var.ty;
                    let data_ty = match var.ty {
                        VarType::Bool => &VarType::U8,
                        _ => &var.ty,
                    };

                    let spirv_ty = self.spirv_ty(ty);
                    let spirv_data_ty = self.spirv_ty(data_ty);
                    let ptr_ty =
                        self.type_pointer(None, spirv::StorageClass::StorageBuffer, spirv_data_ty);
                    let int_ty = self.type_int(32, 0);
                    let buffer_idx = self.constant_bit32(int_ty, buffer_idx);
                    let elem = self.constant_bit32(int_ty, 0);

                    let buffer_array = self.get_buffer_array(&ty);
                    // let src = self.reg(src);
                    let idx = self.reg(idx);

                    // We do not need variables if the gather operation is not conditioned.
                    if let Some(cond) = cond {
                        let cond = self.reg(*cond);

                        let res_var_ty =
                            self.type_pointer(None, spirv::StorageClass::Function, spirv_ty);

                        let result_false = self.constant_null(spirv_ty);
                        let mut result_true = 0;

                        // TODO: find out how to get at label of current block
                        let prev_label = self.id();
                        self.branch(prev_label).unwrap();
                        self.begin_block(Some(prev_label)).unwrap();

                        let true_label = self.if_block(cond, |s| {
                            let ptr = s
                                .access_chain(ptr_ty, None, buffer_array, [buffer_idx, elem, idx])
                                .unwrap();

                            let res = match var.ty {
                                VarType::Bool => {
                                    let bool_ty = s.type_bool();
                                    let u8_ty = s.type_int(8, 0);
                                    let u8_0 = s.constant_bit32(u8_ty, 0);
                                    let data =
                                        s.load(spirv_data_ty, None, ptr, None, None).unwrap();
                                    s.i_not_equal(bool_ty, None, data, u8_0).unwrap()
                                }
                                _ => s.load(spirv_data_ty, None, ptr, None, None).unwrap(),
                            };
                            result_true = res;
                        })?;
                        self.phi(
                            spirv_ty,
                            None,
                            [(result_true, true_label), (result_false, prev_label)],
                        )?
                    } else {
                        let ptr =
                            self.access_chain(ptr_ty, None, buffer_array, [buffer_idx, elem, idx])?;

                        match var.ty {
                            VarType::Bool => {
                                let bool_ty = self.type_bool();
                                let u8_ty = self.type_int(8, 0);
                                let u8_0 = self.constant_bit32(u8_ty, 0);
                                let data = self.load(spirv_data_ty, None, ptr, None, None)?;
                                self.i_not_equal(bool_ty, None, data, u8_0)?
                            }
                            _ => self.load(spirv_data_ty, None, ptr, None, None)?,
                        }
                    }
                }
                KernelOp::Index => {
                    let u32_ty = self.type_int(32, 0);
                    let ptr_ty = self.type_pointer(None, spirv::StorageClass::Input, u32_ty);
                    let u32_0 = self.constant_bit32(u32_ty, 0);
                    let ptr = self.access_chain(ptr_ty, None, global_invocation_id, [u32_0])?;

                    self.load(u32_ty, None, ptr, None, None)?
                }
                KernelOp::Literal => {
                    let ty = self.spirv_ty(&var.ty);
                    match &var.ty {
                        VarType::Bool => {
                            if var.data == 0 {
                                self.constant_false(ty)
                            } else {
                                self.constant_true(ty)
                            }
                        }
                        VarType::I8
                        | VarType::U8
                        | VarType::I16
                        | VarType::U16
                        | VarType::I32
                        | VarType::U32
                        | VarType::I64
                        | VarType::U64 => self.constant_bit64(ty, var.data),
                        VarType::F16 => {
                            self.constant_bit32(ty, unsafe { *(&var.data as *const _ as *const _) })
                        }
                        VarType::F32 => {
                            self.constant_bit32(ty, unsafe { *(&var.data as *const _ as *const _) })
                        }
                        VarType::F64 => {
                            self.constant_bit64(ty, unsafe { *(&var.data as *const _ as *const _) })
                        }
                        _ => todo!(),
                    }
                }
                _ => 0,
            };
            if !self.spriv_regs.contains_key(&varid) {
                self.spriv_regs.insert(varid, res);
            }
        }
        Ok(())
    }
}
impl SpirvBuilder {
    fn bop(
        &mut self,
        op: Bop,
        result_type: &'static VarType,
        src_type: &'static VarType,
        lhs: u32,
        rhs: u32,
    ) -> Result<u32, dr::Error> {
        let spv_type = self.spirv_ty(result_type);
        match src_type {
            VarType::Bool => self.bool_bop(op, spv_type, lhs, rhs),
            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                self.u_bop(op, spv_type, lhs, rhs)
            }
            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                self.s_bop(op, spv_type, lhs, rhs)
            }
            VarType::F16 | VarType::F32 | VarType::F64 => self.f_bop(op, spv_type, lhs, rhs),
            // NOTE: this is a bit of a cheat, but we can change the src type to select the correct
            // binary op variant
            VarType::Vec { ty, .. } => self.bop(op, result_type, ty, lhs, rhs),
            _ => todo!(),
        }
    }
    fn bool_bop(
        &mut self,
        op: Bop,
        result_type: u32,
        lhs: u32,
        rhs: u32,
    ) -> Result<u32, dr::Error> {
        match op {
            Bop::And => self.logical_and(result_type, None, lhs, rhs),
            Bop::Or => self.logical_or(result_type, None, lhs, rhs),
            Bop::Xor => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                let n_rhs = self.logical_not(result_type, None, rhs)?;
                let t0 = self.logical_and(result_type, None, n_lhs, rhs)?;
                let t1 = self.logical_and(result_type, None, lhs, n_rhs)?;
                self.logical_or(result_type, None, t0, t1)
            }
            Bop::Eq => self.logical_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.logical_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                self.logical_and(result_type, None, n_lhs, rhs)
            }
            Bop::Le => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                let n_rhs = self.logical_not(result_type, None, rhs)?;
                let t0 = self.logical_and(result_type, None, n_lhs, rhs)?;
                let t1 = self.logical_and(result_type, None, lhs, rhs)?;
                let t2 = self.logical_and(result_type, None, n_lhs, n_rhs)?;

                let t3 = self.logical_or(result_type, None, t0, t1)?;
                self.logical_or(result_type, None, t3, t2)
            }
            Bop::Gt => {
                let n_rhs = self.logical_not(result_type, None, lhs)?;
                self.logical_and(result_type, None, lhs, n_rhs)
            }
            Bop::Ge => {
                let n_lhs = self.logical_not(result_type, None, lhs)?;
                let n_rhs = self.logical_not(result_type, None, rhs)?;
                let t0 = self.logical_and(result_type, None, lhs, n_rhs)?;
                let t1 = self.logical_and(result_type, None, lhs, rhs)?;
                let t2 = self.logical_and(result_type, None, n_lhs, n_rhs)?;

                let t3 = self.logical_or(result_type, None, t0, t1)?;
                self.logical_or(result_type, None, t3, t2)
            }
            _ => todo!(),
        }
    }
    fn u_bop(&mut self, op: Bop, result_type: u32, lhs: u32, rhs: u32) -> Result<u32, dr::Error> {
        match op {
            Bop::Add => self.i_add(result_type, None, lhs, rhs),
            Bop::Sub => self.i_sub(result_type, None, lhs, rhs),
            Bop::Mul => self.i_mul(result_type, None, lhs, rhs),
            Bop::Div => self.u_div(result_type, None, lhs, rhs),
            Bop::Modulus => self.u_mod(result_type, None, lhs, rhs),
            Bop::Min => glsl_ext!(self, result_type, UMin, lhs, rhs),
            Bop::Max => glsl_ext!(self, result_type, UMax, lhs, rhs),
            Bop::And => self.bitwise_and(result_type, None, lhs, rhs),
            Bop::Or => self.bitwise_or(result_type, None, lhs, rhs),
            Bop::Xor => self.bitwise_xor(result_type, None, lhs, rhs),
            Bop::Shl => self.shift_left_logical(result_type, None, lhs, rhs),
            Bop::Shr => self.shift_right_logical(result_type, None, lhs, rhs),
            Bop::Eq => self.i_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.i_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => self.u_less_than(result_type, None, lhs, rhs),
            Bop::Le => self.u_less_than_equal(result_type, None, lhs, rhs),
            Bop::Gt => self.u_greater_than(result_type, None, lhs, rhs),
            Bop::Ge => self.u_greater_than_equal(result_type, None, lhs, rhs),
        }
    }
    fn s_bop(&mut self, op: Bop, result_type: u32, lhs: u32, rhs: u32) -> Result<u32, dr::Error> {
        match op {
            Bop::Add => self.i_add(result_type, None, lhs, rhs),
            Bop::Sub => self.i_sub(result_type, None, lhs, rhs),
            Bop::Mul => self.i_mul(result_type, None, lhs, rhs),
            Bop::Div => self.s_div(result_type, None, lhs, rhs),
            Bop::Modulus => self.s_mod(result_type, None, lhs, rhs),
            Bop::Min => glsl_ext!(self, result_type, SMin, lhs, rhs),
            Bop::Max => glsl_ext!(self, result_type, SMax, lhs, rhs),
            Bop::And => self.bitwise_and(result_type, None, lhs, rhs),
            Bop::Or => self.bitwise_or(result_type, None, lhs, rhs),
            Bop::Xor => self.bitwise_xor(result_type, None, lhs, rhs),
            Bop::Shl => self.shift_left_logical(result_type, None, lhs, rhs),
            Bop::Shr => self.shift_right_logical(result_type, None, lhs, rhs),
            Bop::Eq => self.i_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.i_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => self.s_less_than(result_type, None, lhs, rhs),
            Bop::Le => self.s_less_than_equal(result_type, None, lhs, rhs),
            Bop::Gt => self.s_greater_than(result_type, None, lhs, rhs),
            Bop::Ge => self.s_greater_than_equal(result_type, None, lhs, rhs),
        }
    }
    fn f_bop(&mut self, op: Bop, result_type: u32, lhs: u32, rhs: u32) -> Result<u32, dr::Error> {
        match op {
            Bop::Add => self.f_add(result_type, None, lhs, rhs),
            Bop::Sub => self.f_sub(result_type, None, lhs, rhs),
            Bop::Mul => self.f_mul(result_type, None, lhs, rhs),
            Bop::Div => self.f_div(result_type, None, lhs, rhs),
            Bop::Modulus => self.f_mod(result_type, None, lhs, rhs),
            Bop::Min => glsl_ext!(self, result_type, FMin, lhs, rhs),
            Bop::Max => glsl_ext!(self, result_type, FMax, lhs, rhs),
            Bop::Eq => self.f_ord_equal(result_type, None, lhs, rhs),
            Bop::Neq => self.f_ord_not_equal(result_type, None, lhs, rhs),
            Bop::Lt => self.f_ord_less_than(result_type, None, lhs, rhs),
            Bop::Le => self.f_ord_less_than_equal(result_type, None, lhs, rhs),
            Bop::Gt => self.f_ord_greater_than(result_type, None, lhs, rhs),
            Bop::Ge => self.f_ord_less_than_equal(result_type, None, lhs, rhs),
            _ => todo!(),
        }
    }
}
