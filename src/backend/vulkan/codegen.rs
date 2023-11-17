use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::backend::vulkan::glslext::GLSL450Instruction;
use crate::ir::{Bop, Op, VarId, IR};
use crate::vartype::VarType;
use rspirv::binary::{Assemble, Disassemble};
use rspirv::{dr, spirv};

// fn ty(ty: &VarType) ->

pub fn assemble_trace(trace: &IR, entry_point: &str) -> Result<Vec<u32>, dr::Error> {
    let mut b = SpirvBuilder::default();
    b.assemble(trace, entry_point)?;

    let module = b.module();
    print!("{}", module.disassemble());
    Ok(module.assemble())
}
fn isfloat(ty: &VarType) -> bool {
    match ty {
        VarType::F32 | VarType::F64 => true,
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

#[derive(Default)]
/// A wrapper arround `dr::Builder` to generate spriv code.
///
/// * `b`: Builder from `rspriv`
/// * `spirv_vars`: Mapping from VarId -> Spirv Id
/// * `glsl_ext`: GLSL_EXT libarary loaded by default
struct SpirvBuilder {
    b: dr::Builder,
    // spirv_vars: Vec<u32>,
    spriv_regs: HashMap<VarId, u32>,
    glsl_ext: u32,
    function_block: usize,
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
    pub fn assemble(&mut self, ir: &IR, entry_point: &str) -> Result<(), dr::Error> {
        // let param_layout = ParamLayout::generate(trace);
        // dbg!(&param_layout);

        self.id();

        // self.acquire_ids(ir);

        self.set_version(1, 5);

        self.capability(spirv::Capability::Shader);
        self.capability(spirv::Capability::Int8);
        self.capability(spirv::Capability::Int16);
        self.capability(spirv::Capability::Int64);
        self.capability(spirv::Capability::Float16);
        self.capability(spirv::Capability::Float64);

        self.capability(spirv::Capability::StorageUniformBufferBlock16);

        self.glsl_ext = self.ext_inst_import("GLSL.std.450");
        self.extension("SPV_KHR_16bit_storage");
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

        let storage_vars = self.assemble_storage_vars(ir);
        let samplers = self.assemble_samplers(ir);

        let func = self.begin_function(
            void,
            None,
            spirv::FunctionControl::DONT_INLINE | spirv::FunctionControl::CONST,
            voidf,
        )?;

        self.begin_block(None)?;
        self.function_block = self.selected_block().unwrap();

        self.assemble_vars(ir, global_invocation_id)?;

        self.ret()?;
        self.end_function()?;

        let interface = [global_invocation_id]
            .into_iter()
            .chain(storage_vars.into_iter())
            .chain(samplers.into_iter())
            .collect::<Vec<_>>();

        self.entry_point(
            spirv::ExecutionModel::GLCompute,
            func,
            entry_point,
            interface,
        );
        //OpEntryPoint GLCompute %main "main" %test %gl_GlobalInvocationID %test2
        self.execution_mode(func, spirv::ExecutionMode::LocalSize, [1, 1, 1]);

        Ok(())
    }
    // fn acquire_ids(&mut self, trace: &IR) {
    //     let vars = trace.vars.iter().map(|_| self.id()).collect::<Vec<_>>();
    //     self.spirv_vars = vars;
    // }
    fn reg(&self, id: VarId) -> u32 {
        self.spriv_regs[&id]
    }
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
    fn if_block<T, F: FnOnce(&mut Self) -> Result<T, dr::Error>>(
        &mut self,
        cond: u32,
        f: F,
    ) -> Result<T, dr::Error> {
        let ty_bool = self.type_bool();
        let bool_true = self.constant_true(ty_bool);

        let cond = self.logical_equal(ty_bool, None, cond, bool_true)?;

        let start_label = self.id();
        let end_label = self.id();

        // According to spirv OpSelectionMerge should be second to last
        // instruction in block. Rspirv however ends block with
        // selection_merge. Therefore, we insert the instruction by hand.
        self.b
            .insert_into_block(
                rspirv::dr::InsertPoint::End,
                rspirv::dr::Instruction::new(
                    spirv::Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(end_label),
                        rspirv::dr::Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
            )
            .unwrap();
        self.branch_conditional(cond, start_label, end_label, [])
            .unwrap();

        self.begin_block(Some(start_label))?;

        let res = f(self)?;

        self.branch(end_label).unwrap();

        self.begin_block(Some(end_label))?;
        Ok(res)
    }
    fn spirv_ty(&mut self, ty: &VarType) -> u32 {
        match ty {
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
            VarType::F32 => self.type_float(32),
            VarType::F64 => self.type_float(64),
            VarType::Vec { ty, num } => {
                let ty = self.spirv_ty(&ty);
                self.type_vector(ty, *num as _)
            }
            VarType::Struct { tys } => {
                let spv_tys = tys.iter().map(|ty| self.spirv_ty(ty)).collect::<Vec<_>>();
                let struct_ty = self.type_struct(spv_tys);
                for i in 0..tys.len() {
                    let offset = ty.offset(i);
                    self.member_decorate(
                        struct_ty,
                        offset as _,
                        spirv::Decoration::Offset,
                        [dr::Operand::LiteralInt32(offset as _)],
                    );
                }
                struct_ty
            }
        }
    }
    fn assemble_samplers(&mut self, ir: &IR) -> Vec<u32> {
        ir.var_ids()
            .filter_map(|varid| {
                let var = ir.var(varid);
                match var.op {
                    Op::TextureRef => {
                        let ty = self.spirv_ty(&var.ty);
                        let ty_image = self.type_image(
                            ty,
                            spirv::Dim::Dim2D,
                            0,
                            0,
                            0,
                            1,
                            spirv::ImageFormat::Unknown,
                            None,
                        );
                        let ty_sampled_image = self.type_sampled_image(ty_image);

                        let ty_int = self.type_int(32, 0);
                        let length = self.constant_u32(ty_int, ir.n_textures as _);

                        let ty_array = self.type_array(ty_sampled_image, length);
                        let ty_ptr =
                            self.type_pointer(None, spirv::StorageClass::UniformConstant, ty_array);

                        let res =
                            self.variable(ty_ptr, None, spirv::StorageClass::UniformConstant, None);

                        self.decorate(
                            res,
                            spirv::Decoration::DescriptorSet,
                            [dr::Operand::LiteralInt32(0)],
                        );
                        self.decorate(
                            res,
                            spirv::Decoration::Binding,
                            [dr::Operand::LiteralInt32(1)],
                        );

                        self.spriv_regs.insert(varid, res);
                        dbg!(res);
                        Some(res)
                    }
                    _ => None,
                }
            })
            .collect()
    }
    // TODO: Spirv Builder
    fn assemble_storage_vars(&mut self, ir: &IR) -> Vec<u32> {
        ir.var_ids()
            .filter_map(|varid| {
                let var = ir.var(varid);
                match var.op {
                    Op::BufferRef => {
                        let ty = match var.ty {
                            VarType::Bool => &VarType::U8,
                            _ => &var.ty,
                        };
                        let spriv_ty = self.spirv_ty(&ty);
                        let u32_ty = self.type_int(32, 0);
                        let array_len = self.constant_u32(u32_ty, ir.n_buffers as _);
                        let rta_ty = self.type_runtime_array(spriv_ty);
                        let struct_ty = self.type_struct([rta_ty]);
                        let array_ty = self.type_array(struct_ty, array_len);

                        self.decorate(struct_ty, spirv::Decoration::Block, []);
                        self.member_decorate(
                            struct_ty,
                            0,
                            spirv::Decoration::Offset,
                            [dr::Operand::LiteralInt32(0)],
                        );
                        self.decorate(
                            rta_ty,
                            rspirv::spirv::Decoration::ArrayStride,
                            [dr::Operand::LiteralInt32(ty.size() as _)],
                        );

                        let ptr_ty =
                            self.type_pointer(None, spirv::StorageClass::StorageBuffer, array_ty);

                        // let dst = self.get(varid);

                        let res =
                            self.variable(ptr_ty, None, spirv::StorageClass::StorageBuffer, None);

                        self.decorate(
                            res,
                            spirv::Decoration::DescriptorSet,
                            [dr::Operand::LiteralInt32(0)],
                        );
                        self.decorate(
                            res,
                            spirv::Decoration::Binding,
                            [dr::Operand::LiteralInt32(0)],
                        );

                        self.spriv_regs.insert(varid, res);
                        dbg!(res);
                        Some(res)
                    }
                    _ => None,
                }
            })
            .collect()
    }

    fn assemble_vars(&mut self, ir: &IR, global_invocation_id: u32) -> Result<(), dr::Error> {
        macro_rules! bop {
            ($bop:ident: $int:ident, $float:ident) => {
                crate::op::Bop::$bop => {
                    if isint(&var.ty) {
                        self.$int(ty, Some(dst), lhs, rhs)?;
                    } else if isfloat(&var.ty) {
                        self.$float(ty, Some(dst), lhs, rhs)?;
                    } else {
                        todo!()
                    }
                }
            };
        }
        macro_rules! glsl_ext {
            ($self:ident; $dst:ident : $ty:ident = $ext:ident, $($operand:expr),*) => {
                let ext = $self.glsl_ext;
                $self.ext_inst(
                    $ty,
                    Some($dst),
                    ext,
                    GLSL450Instruction::$ext as _,
                    [$(dr::Operand::IdRef($operand)),*],
                )?;
            };
        }
        for varid in ir.var_ids() {
            let var = ir.var(varid);
            let deps = ir.deps(varid);
            let res = match var.op {
                Op::Nop => 0,
                Op::Bop(bop) => {
                    let res = self.id();
                    let lhs = self.reg(deps[0]);
                    let rhs = self.reg(deps[1]);
                    let ty = self.spirv_ty(&var.ty);
                    match bop {
                        Bop::Add => {
                            if isint(&var.ty) {
                                self.i_add(ty, Some(res), lhs, rhs)?;
                            } else if isfloat(&var.ty) {
                                self.f_add(ty, Some(res), lhs, rhs)?;
                            } else {
                                todo!()
                            }
                        }
                        Bop::Sub => {
                            if isint(&var.ty) {
                                self.i_sub(ty, Some(res), lhs, rhs)?;
                            } else if isfloat(&var.ty) {
                                self.f_sub(ty, Some(res), lhs, rhs)?;
                            } else {
                                todo!()
                            }
                        }
                        Bop::Mul => {
                            if isint(&var.ty) {
                                self.i_mul(ty, Some(res), lhs, rhs)?;
                            } else if isfloat(&var.ty) {
                                self.f_mul(ty, Some(res), lhs, rhs)?;
                            } else {
                                todo!()
                            }
                        }
                        Bop::Div => match var.ty {
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                self.s_div(ty, Some(res), lhs, rhs)?;
                            }
                            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                self.u_div(ty, Some(res), lhs, rhs)?;
                            }
                            VarType::F32 | VarType::F64 => {
                                self.f_div(ty, Some(res), lhs, rhs)?;
                            }
                            _ => todo!(),
                        },
                        Bop::Min => match var.ty {
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                glsl_ext!(self; res: ty = SMin, lhs, rhs);
                            }
                            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                glsl_ext!(self; res: ty = UMin, lhs, rhs);
                            }
                            VarType::F32 | VarType::F64 => {
                                glsl_ext!(self; res: ty = FMin, lhs, rhs);
                            }
                            _ => todo!(),
                        },
                        Bop::Max => match var.ty {
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                glsl_ext!(self; res: ty = SMax, lhs, rhs);
                            }
                            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                glsl_ext!(self; res: ty = UMax, lhs, rhs);
                            }
                            VarType::F32 | VarType::F64 => {
                                glsl_ext!(self; res: ty = FMax, lhs, rhs);
                            }
                            _ => todo!(),
                        },
                    };
                    res
                }
                Op::Construct => {
                    let ty = self.spirv_ty(&var.ty);
                    let deps = deps.iter().map(|id| self.reg(*id)).collect::<Vec<_>>();
                    self.composite_construct(ty, None, deps)?
                }
                Op::Extract(elem) => {
                    let ty = self.spirv_ty(&ir.var(varid).ty);
                    let src = self.reg(deps[0]);
                    self.composite_extract(ty, None, src, [elem as u32])?
                }
                Op::TexLookup => {
                    let img = deps[0];
                    let coord = self.reg(deps[1]);

                    let img_idx = ir.var(img).data;

                    let ty = self.spirv_ty(ir.var_ty(img));
                    let ty_image = self.type_image(
                        ty,
                        spirv::Dim::Dim2D,
                        0,
                        0,
                        0,
                        1,
                        spirv::ImageFormat::Unknown,
                        None,
                    );
                    let ty_sampled_image = self.type_sampled_image(ty_image);
                    let ptr_ty = self.type_pointer(
                        None,
                        spirv::StorageClass::UniformConstant,
                        ty_sampled_image,
                    );

                    let int_ty = self.type_int(32, 0);
                    let img_idx = self.constant_u32(int_ty, img_idx as _);

                    let img = self.reg(img);
                    let ptr = self.access_chain(ptr_ty, None, img, [img_idx])?;

                    let ty_v4 = self.type_vector(ty, 4);

                    let float_ty = self.type_float(32);
                    let float_0 = self.constant_f32(float_ty, 0.);

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
                Op::Scatter => {
                    let dst = deps[0];
                    let src = deps[1];
                    let idx = deps[2];
                    let cond = deps.get(3);
                    let ty = &ir.var(src).ty;
                    let data_ty = match ty {
                        VarType::Bool => &VarType::U8,
                        _ => &ir.var(src).ty,
                    };

                    let buffer_idx = ir.var(dst).data;

                    let spriv_ty = self.spirv_ty(data_ty);
                    let ptr_ty =
                        self.type_pointer(None, spirv::StorageClass::StorageBuffer, spriv_ty);
                    let int_ty = self.type_int(32, 0);
                    let buffer = self.constant_u32(int_ty, buffer_idx as _);
                    let elem = self.constant_u32(int_ty, 0);

                    let dst = self.reg(dst);
                    let idx = self.reg(idx);
                    let src = self.reg(src);

                    // Don't need to condition the scatter if that's not neccesarry
                    // TODO: unify conditioned and uncoditioned part
                    if let Some(cond) = cond {
                        assert_eq!(ir.var(*cond).ty, VarType::Bool);

                        let cond = self.reg(*cond);
                        self.if_block(cond, |s| {
                            let ptr = s.access_chain(ptr_ty, None, dst, [buffer, elem, idx])?;

                            match ty {
                                VarType::Bool => {
                                    let u8_ty = s.type_int(8, 0);
                                    let u8_0 = s.constant_u32(u8_ty, 0);
                                    let u8_1 = s.constant_u32(u8_ty, 1);

                                    let data = s.select(u8_ty, None, src, u8_1, u8_0)?;
                                    dbg!(data);
                                    s.store(ptr, data, None, None)?;
                                }
                                _ => {
                                    s.store(ptr, src, None, None)?;
                                }
                            };

                            Ok(())
                        })?;
                    } else {
                        let ptr = self.access_chain(ptr_ty, None, dst, [buffer, elem, idx])?;

                        match ty {
                            VarType::Bool => {
                                let u8_ty = self.type_int(8, 0);
                                let u8_0 = self.constant_u32(u8_ty, 0);
                                let u8_1 = self.constant_u32(u8_ty, 1);

                                let data = self.select(u8_ty, None, src, u8_1, u8_0)?;
                                dbg!(data);
                                self.store(ptr, data, None, None)?;
                            }
                            _ => {
                                self.store(ptr, src, None, None)?;
                            }
                        };
                    }
                    0
                }
                Op::Gather => {
                    let src = deps[0];
                    let idx = deps[1];
                    let cond = deps.get(2);
                    let buffer_idx = ir.var(src).data;

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
                    let buffer_idx = self.constant_u32(int_ty, buffer_idx as _);
                    let elem = self.constant_u32(int_ty, 0);

                    let src = self.reg(src);
                    let idx = self.reg(idx);

                    // We do not need variables if the gather operation is not conditioned.
                    if let Some(cond) = cond {
                        let cond = self.reg(*cond);

                        let res_var_ty =
                            self.type_pointer(None, spirv::StorageClass::Function, spirv_ty);
                        let res_var = self.id();
                        // Insert a variable at the beginning of the function (bit hacky)
                        self.with_function(|s| {
                            s.insert_into_block(
                                dr::InsertPoint::Begin,
                                dr::Instruction::new(
                                    spirv::Op::Variable,
                                    Some(res_var_ty),
                                    Some(res_var),
                                    vec![dr::Operand::StorageClass(spirv::StorageClass::Function)],
                                ),
                            )?;
                            Ok(())
                        })?;

                        self.if_block(cond, |s| {
                            let ptr = s.access_chain(ptr_ty, None, src, [buffer_idx, elem, idx])?;

                            let res = match var.ty {
                                VarType::Bool => {
                                    let bool_ty = s.type_bool();
                                    let u8_ty = s.type_int(8, 0);
                                    let u8_0 = s.constant_u32(u8_ty, 0);
                                    let data = s.load(spirv_data_ty, None, ptr, None, None)?;
                                    s.i_not_equal(bool_ty, None, data, u8_0)?
                                }
                                _ => s.load(spirv_data_ty, None, ptr, None, None)?,
                            };

                            s.store(res_var, res, None, None)?;

                            Ok(())
                        })?;
                        self.load(spirv_ty, None, res_var, None, None)?
                    } else {
                        let ptr = self.access_chain(ptr_ty, None, src, [buffer_idx, elem, idx])?;

                        match var.ty {
                            VarType::Bool => {
                                let bool_ty = self.type_bool();
                                let u8_ty = self.type_int(8, 0);
                                let u8_0 = self.constant_u32(u8_ty, 0);
                                let data = self.load(spirv_data_ty, None, ptr, None, None)?;
                                self.i_not_equal(bool_ty, None, data, u8_0)?
                            }
                            _ => self.load(spirv_data_ty, None, ptr, None, None)?,
                        }
                    }
                }
                Op::Index => {
                    let u32_ty = self.type_int(32, 0);
                    let ptr_ty = self.type_pointer(None, spirv::StorageClass::Input, u32_ty);
                    let u32_0 = self.constant_u32(u32_ty, 0);
                    let ptr = self.access_chain(ptr_ty, None, global_invocation_id, [u32_0])?;

                    self.load(u32_ty, None, ptr, None, None)?
                }
                Op::Literal => {
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
                        | VarType::U64
                        | VarType::F32
                        | VarType::F64 => self.constant_u32(ty, var.data as _),
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
