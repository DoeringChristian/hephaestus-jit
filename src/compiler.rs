use crate::ftrace::{self, FTrace};
use crate::ir::{self, IR};
use crate::op::{KernelOp, Op};
use crate::vartype::{self, AsVarType};
use std::collections::HashMap;
use std::ops::Range;

#[derive(Debug, Default)]
pub struct Compiler {
    pub ir: IR,
    visited: HashMap<ftrace::Var, ir::VarId>,
    trivial: HashMap<ir::Var, ir::VarId>,
    id2buffer: HashMap<ftrace::Var, usize>,
    pub buffers: Vec<ftrace::Var>,
    id2texture: HashMap<ftrace::Var, usize>,
    pub textures: Vec<ftrace::Var>,
    id2accel: HashMap<ftrace::Var, usize>,
    pub accels: Vec<ftrace::Var>,
}

impl Compiler {
    pub fn compile(&mut self, trace: &FTrace, ids: &[ftrace::Var]) {
        for id in ids {
            let src = self.collect(trace, *id);

            let var = trace.entry(*id);
            // let scope = var.scope;
            if var.ty.size() == 0 {
                continue;
            }

            let buffer_id = self.push_buffer(*id);
            // TODO: maybe some optimization regarding trivial vars
            let dst = self.push_var(
                ir::Var {
                    op: KernelOp::BufferRef,
                    ty: var.ty,
                    data: buffer_id as _,
                    // scope,
                    ..Default::default()
                },
                [],
            );
            let idx = self.push_var(
                ir::Var {
                    op: KernelOp::Index,
                    ty: u32::var_ty(),
                    // scope,
                    ..Default::default()
                },
                [],
            );
            self.push_var(
                ir::Var {
                    op: KernelOp::Scatter,
                    ty: vartype::void(),
                    // scope,
                    ..Default::default()
                },
                [dst, src, idx],
            );
        }
        // self.ir.scope_sort();
        self.ir.n_buffers = self.buffers.len();
        self.ir.n_textures = self.textures.len();
        self.ir.n_accels = self.accels.len();
    }
    pub fn collect(&mut self, trace: &ftrace::FTrace, id: ftrace::Var) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.entry(id);
        // let scope = var.scope;

        let internal_id = match var.op {
            Op::Ref { .. } => {
                // When we hit a ref, we just load it as a ref
                self.collect_data(trace, trace.entry(id).deps[0])
            }
            Op::Buffer => {
                // When we hit a buffer directly we want to access the elements directly
                let data = self.collect_data(trace, id);
                let idx = self.push_var(
                    ir::Var {
                        op: KernelOp::Index,
                        ty: u32::var_ty(),
                        // scope,
                        ..Default::default()
                    },
                    [],
                );
                self.push_var(
                    ir::Var {
                        op: KernelOp::Gather,
                        ty: var.ty,
                        // scope,
                        ..Default::default()
                    },
                    [data, idx],
                )
            }
            Op::KernelOp(kop) => match kop {
                KernelOp::Literal => self.push_var(
                    ir::Var {
                        op: KernelOp::Literal,
                        ty: var.ty,
                        data: var.literal,
                        // scope,
                        ..Default::default()
                    },
                    [],
                ),
                _ => {
                    let deps = trace
                        .entry(id)
                        .deps
                        .iter()
                        .map(|id| self.collect(trace, *id))
                        .collect::<Vec<_>>();
                    self.push_var(
                        ir::Var {
                            op: kop,
                            ty: var.ty,
                            // scope,
                            ..Default::default()
                        },
                        deps,
                    )
                }
            },
            _ => todo!(), // Op::Buffer => self.collect_data(trace, id),
        };
        self.visited.insert(id, internal_id);
        internal_id
    }
    pub fn collect_data(&mut self, trace: &ftrace::FTrace, id: ftrace::Var) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.entry(id);

        // NOTE: We can leave the scope for references at it's default since they can always be put
        // in front.

        match trace.entry(id).op.resulting_op() {
            Op::Buffer => {
                let buffer_id = self.push_buffer(id);
                self.push_var(
                    ir::Var {
                        op: KernelOp::BufferRef,
                        ty: var.ty,
                        data: buffer_id as _,
                        ..Default::default()
                    },
                    [],
                )
            }
            Op::Texture => {
                let texture_id = self.push_texture(id);
                let dim = trace.entry(id).extent.texture_dim();
                self.push_var(
                    ir::Var {
                        op: KernelOp::TextureRef { dim },
                        ty: var.ty,
                        data: texture_id as _,
                        ..Default::default()
                    },
                    [],
                )
            }
            Op::Accel => {
                let accel_id = self.push_accel(id);
                self.push_var(
                    ir::Var {
                        op: KernelOp::AccelRef,
                        ty: var.ty,
                        data: accel_id as _,
                        ..Default::default()
                    },
                    [],
                )
            }
            _ => todo!(),
        }
    }
    pub fn push_buffer(&mut self, id: ftrace::Var) -> usize {
        *self.id2buffer.entry(id).or_insert_with(|| {
            let buffer_id = self.buffers.len();
            self.buffers.push(id);
            buffer_id
        })
    }
    pub fn push_texture(&mut self, id: ftrace::Var) -> usize {
        *self.id2texture.entry(id).or_insert_with(|| {
            let texture_id = self.textures.len();
            self.textures.push(id);
            texture_id
        })
    }
    pub fn push_accel(&mut self, id: ftrace::Var) -> usize {
        *self.id2accel.entry(id).or_insert_with(|| {
            let accel_id = self.accels.len();
            self.accels.push(id);
            accel_id
        })
    }
    pub fn push_var<I>(&mut self, mut var: ir::Var, deps: I) -> ir::VarId
    where
        I: IntoIterator<Item = ir::VarId>,
        I::IntoIter: ExactSizeIterator,
    {
        // Trivial vars (without dependencies) can be optimized out using a hash map.
        // Their scope is always zero, as they can be put at the start of the kernel.
        let deps = deps.into_iter();
        let trivial = deps.len() == 0;

        if trivial {
            // var.scope = Default::default();
            var.deps = Default::default();
            if self.trivial.contains_key(&var) {
                return self.trivial[&var];
            }
        }
        let id = self.ir.push_var(var, deps);

        if trivial {
            self.trivial.insert(var, id);
        }

        id
    }
    pub fn clear(&mut self) {
        self.ir.clear();
        self.trivial.clear();
        self.visited.clear();
        self.id2buffer.clear();
        self.buffers.clear();
        self.id2texture.clear();
        self.textures.clear();
        self.id2accel.clear();
        self.accels.clear();
    }
}
