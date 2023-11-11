use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::backend::vulkan::glslext::GLSL450Instruction;
use crate::ir::{VarId, IR};
use crate::op::Op;
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
struct SpirvBuilder {
    b: dr::Builder,
    spirv_vars: Vec<u32>,
    glsl_ext: u32,
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
    pub fn assemble(&mut self, trace: &IR, entry_point: &str) -> Result<(), dr::Error> {
        // let param_layout = ParamLayout::generate(trace);
        // dbg!(&param_layout);

        self.id();

        self.acquire_ids(trace);

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

        let storage_vars = self.assemble_storage_vars(trace);

        let func = self.begin_function(
            void,
            None,
            spirv::FunctionControl::DONT_INLINE | spirv::FunctionControl::CONST,
            voidf,
        )?;

        self.begin_block(None)?;

        self.assemble_vars(trace, global_invocation_id)?;

        self.ret()?;
        self.end_function()?;

        let interface = [global_invocation_id]
            .into_iter()
            .chain(storage_vars.into_iter())
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
    fn acquire_ids(&mut self, trace: &IR) {
        let vars = trace.vars.iter().map(|_| self.id()).collect::<Vec<_>>();
        self.spirv_vars = vars;
    }
    fn get(&self, id: VarId) -> u32 {
        self.spirv_vars[id.0]
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
    // TODO: Spirv Builder
    fn assemble_storage_vars(&mut self, ir: &IR) -> Vec<u32> {
        ir.var_ids()
            .filter_map(|varid| {
                let var = ir.var(varid);
                match var.op {
                    Op::Buffer => {
                        let ty = self.spirv_ty(&var.ty);
                        let u32_ty = self.type_int(32, 0);
                        let array_len = self.constant_u32(u32_ty, ir.n_buffers as _);
                        let rta_ty = self.type_runtime_array(ty);
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
                            [dr::Operand::LiteralInt32(var.ty.size() as _)],
                        );

                        let ptr_ty =
                            self.type_pointer(None, spirv::StorageClass::StorageBuffer, array_ty);

                        let dst = self.get(varid);

                        self.variable(ptr_ty, Some(dst), spirv::StorageClass::StorageBuffer, None);

                        self.decorate(
                            dst,
                            spirv::Decoration::DescriptorSet,
                            [dr::Operand::LiteralInt32(0)],
                        );
                        self.decorate(
                            dst,
                            spirv::Decoration::Binding,
                            [dr::Operand::LiteralInt32(0)],
                        );
                        Some(dst)
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
            match var.op {
                Op::Nop => {}
                Op::Bop(bop) => {
                    let dst = self.get(varid);
                    let lhs = self.get(deps[0]);
                    let rhs = self.get(deps[1]);
                    let ty = self.spirv_ty(&var.ty);
                    match bop {
                        crate::op::Bop::Add => {
                            if isint(&var.ty) {
                                self.i_add(ty, Some(dst), lhs, rhs)?;
                            } else if isfloat(&var.ty) {
                                self.f_add(ty, Some(dst), lhs, rhs)?;
                            } else {
                                todo!()
                            }
                        }
                        crate::op::Bop::Sub => {
                            if isint(&var.ty) {
                                self.i_sub(ty, Some(dst), lhs, rhs)?;
                            } else if isfloat(&var.ty) {
                                self.f_sub(ty, Some(dst), lhs, rhs)?;
                            } else {
                                todo!()
                            }
                        }
                        crate::op::Bop::Mul => {
                            if isint(&var.ty) {
                                self.i_mul(ty, Some(dst), lhs, rhs)?;
                            } else if isfloat(&var.ty) {
                                self.f_mul(ty, Some(dst), lhs, rhs)?;
                            } else {
                                todo!()
                            }
                        }
                        crate::op::Bop::Div => match var.ty {
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                self.s_div(ty, Some(dst), lhs, rhs)?;
                            }
                            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                self.u_div(ty, Some(dst), lhs, rhs)?;
                            }
                            VarType::F32 | VarType::F64 => {
                                self.f_div(ty, Some(dst), lhs, rhs)?;
                            }
                            _ => todo!(),
                        },
                        crate::op::Bop::Min => match var.ty {
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                glsl_ext!(self; dst: ty = SMin, lhs, rhs);
                            }
                            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                glsl_ext!(self; dst: ty = UMin, lhs, rhs);
                            }
                            VarType::F32 | VarType::F64 => {
                                glsl_ext!(self; dst: ty = FMin, lhs, rhs);
                            }
                            _ => todo!(),
                        },
                        crate::op::Bop::Max => match var.ty {
                            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => {
                                glsl_ext!(self; dst: ty = SMax, lhs, rhs);
                            }
                            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => {
                                glsl_ext!(self; dst: ty = UMax, lhs, rhs);
                            }
                            VarType::F32 | VarType::F64 => {
                                glsl_ext!(self; dst: ty = FMax, lhs, rhs);
                            }
                            _ => todo!(),
                        },
                    }
                }
                Op::Construct => {
                    let dst = self.get(varid);
                    let ty = self.spirv_ty(&var.ty);
                    let deps = deps.iter().map(|id| self.get(*id)).collect::<Vec<_>>();
                    self.composite_construct(ty, Some(dst), deps)?;
                }
                Op::Extract(elem) => {
                    // Store into temporary variable
                    let src_ty = &ir.var(deps[0]).ty;
                    let src_ty = self.spirv_ty(src_ty);
                    let src_ptr_ty = self.type_pointer(None, spirv::StorageClass::Function, src_ty);
                    let src_var =
                        self.variable(src_ptr_ty, None, spirv::StorageClass::Function, None);

                    let src = self.get(deps[0]);
                    self.store(src_var, src, None, None)?;

                    // Use Access chain to get element

                    let dst = self.get(varid);
                    let ty = self.spirv_ty(ir.var_ty(varid));
                    let ty_ptr = self.type_pointer(None, spirv::StorageClass::Function, ty);
                    let u32_ty = self.type_int(32, 0);
                    let elem = self.constant_u32(u32_ty, elem as _);
                    let ptr = self.access_chain(ty_ptr, None, src_var, [elem])?;
                    self.load(ty, Some(dst), ptr, None, None)?;
                }
                Op::Scatter => {
                    let dst = deps[0];
                    let src = deps[1];
                    let idx = deps[2];
                    dbg!(&ir.var(src));
                    dbg!(&ir.var(dst));

                    let buffer_idx = ir.var(dst).data;

                    let ty = self.spirv_ty(ir.var_ty(src));
                    let ptr_ty = self.type_pointer(None, spirv::StorageClass::StorageBuffer, ty);
                    let int_ty = self.type_int(32, 0);
                    let buffer = self.constant_u32(int_ty, buffer_idx as _);
                    let elem = self.constant_u32(int_ty, 0);

                    let dst = self.get(dst);
                    let idx = self.get(idx);
                    let src = self.get(src);

                    let ptr = self.access_chain(ptr_ty, None, dst, [buffer, elem, idx])?;
                    self.store(ptr, src, None, None)?;
                }
                Op::Gather => {
                    let src = deps[0];
                    let idx = deps[1];
                    let buffer_idx = ir.var(src).data;

                    let ty = self.spirv_ty(&var.ty);
                    let ptr_ty = self.type_pointer(None, spirv::StorageClass::StorageBuffer, ty);
                    let int_ty = self.type_int(32, 0);
                    let buffer = self.constant_u32(int_ty, buffer_idx as _);
                    let elem = self.constant_u32(int_ty, 0);

                    let src = self.get(src);
                    let idx = self.get(idx);
                    let dst = self.get(varid);

                    let ptr = self.access_chain(ptr_ty, None, src, [buffer, elem, idx])?;
                    self.load(ty, Some(dst), ptr, None, None)?;
                }
                Op::Index => {
                    let u32_ty = self.type_int(32, 0);
                    let ptr_ty = self.type_pointer(None, spirv::StorageClass::Input, u32_ty);
                    let u32_0 = self.constant_u32(u32_ty, 0);
                    let ptr = self.access_chain(ptr_ty, None, global_invocation_id, [u32_0])?;

                    let dst = self.get(varid);

                    self.load(u32_ty, Some(dst), ptr, None, None)?;
                }
                Op::Literal => {
                    let ty = self.spirv_ty(&var.ty);
                    let c = match &var.ty {
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
                    };
                    self.spirv_vars[varid.0] = c;
                }
                _ => {}
            }
        }
        Ok(())
    }
}
