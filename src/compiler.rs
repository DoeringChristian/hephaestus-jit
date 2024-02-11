use indexmap::{IndexMap, IndexSet};

use crate::ir::{self, IR};
use crate::op::{KernelOp, Op};
use crate::trace::{self, Trace};
use crate::vartype::{self, AsVarType};
use std::collections::HashMap;
use std::ops::Range;

#[derive(Debug, Default)]
pub struct Compiler {
    pub ir: IR,
    visited: HashMap<trace::VarId, ir::VarId>,
    id2buffer: HashMap<trace::VarId, usize>,
    pub buffers: Vec<trace::VarId>,
    id2texture: HashMap<trace::VarId, usize>,
    pub textures: Vec<trace::VarId>,
    id2accel: HashMap<trace::VarId, usize>,
    pub accels: Vec<trace::VarId>,
}

impl Compiler {
    pub fn compile(&mut self, trace: &Trace, ids: &[trace::VarId]) {
        let mut vars = IndexSet::new();

        fn collect(vars: &mut IndexSet<trace::VarId>, trace: &Trace, id: trace::VarId) {
            if vars.contains(&id) {
                return;
            }
            let var = trace.var(id);
            match var.op {
                Op::KernelOp(_) => {
                    for dep in var.deps.iter() {
                        collect(vars, trace, *dep);
                    }
                }
                Op::Buffer | Op::Texture | Op::Accel => {}
                _ => todo!(),
            }
            vars.insert(id);
        }

        // Collect variables in topological order
        for id in ids.iter() {
            collect(&mut vars, trace, *id);
        }
        // Sort by scope (should preserve topological order)
        vars.sort_by(|id0, id1| trace.var(*id0).scope.cmp(&trace.var(*id1).scope));

        let mut idx2id = Vec::with_capacity(vars.len());

        // Generate IR
        for id in vars.iter() {
            let var = trace.var(*id);

            // TODO: handle buffer references differentily to direct buffer accesses
            let id = match var.op {
                Op::Buffer => {
                    // When we hit a buffer directly we want to access the elements
                    let data = self.collect_data(trace, *id);
                    let idx = self.ir.push_var(
                        ir::Var {
                            op: KernelOp::Index,
                            ty: u32::var_ty(),
                            ..Default::default()
                        },
                        [],
                    );
                    self.ir.push_var(
                        ir::Var {
                            op: KernelOp::Gather,
                            ty: var.ty,
                            ..Default::default()
                        },
                        [data, idx],
                    )
                }
                Op::KernelOp(op) => match op {
                    KernelOp::Literal => self.ir.push_var(
                        ir::Var {
                            op: KernelOp::Literal,
                            ty: var.ty,
                            data: var.data.literal().unwrap(),
                            ..Default::default()
                        },
                        [],
                    ),
                    _ => {
                        let deps = trace
                            .deps(*id)
                            .into_iter()
                            .map(|id| idx2id[vars.get_index_of(id).unwrap()])
                            .collect::<Vec<_>>();
                        self.ir.push_var(
                            ir::Var {
                                op,
                                ty: var.ty,
                                ..Default::default()
                            },
                            deps,
                        )
                    }
                },
                _ => todo!(),
            };
            idx2id.push(id);
        }

        // Scatter output variables into buffers
        for id in ids {
            let var = trace.var(*id);
            if var.ty.size() == 0 {
                continue;
            }

            let src = idx2id[vars.get_index_of(id).unwrap()];

            let buffer_id = self.push_buffer(*id);
            let dst = self.ir.push_var(
                ir::Var {
                    op: KernelOp::BufferRef,
                    ty: var.ty,
                    data: buffer_id as _,
                    ..Default::default()
                },
                [],
            );
            let idx = self.ir.push_var(
                ir::Var {
                    op: KernelOp::Index,
                    ty: u32::var_ty(),
                    ..Default::default()
                },
                [],
            );
            self.ir.push_var(
                ir::Var {
                    op: KernelOp::Scatter,
                    ty: vartype::void(),
                    ..Default::default()
                },
                [dst, src, idx],
            );
        }
        self.ir.n_buffers = self.buffers.len();
        self.ir.n_textures = self.textures.len();
        self.ir.n_accels = self.accels.len();
    }
    pub fn collect(&mut self, trace: &trace::Trace, id: trace::VarId) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.var(id);

        let id = match var.op {
            Op::Ref { .. } => {
                // When we hit a ref, we just load it as a ref
                self.collect_data(trace, trace.deps(id)[0])
            }
            Op::Buffer => {
                // When we hit a buffer directly we want to access the elements directly
                let data = self.collect_data(trace, id);
                let idx = self.ir.push_var(
                    ir::Var {
                        op: KernelOp::Index,
                        ty: u32::var_ty(),
                        ..Default::default()
                    },
                    [],
                );
                self.ir.push_var(
                    ir::Var {
                        op: KernelOp::Gather,
                        ty: var.ty,
                        ..Default::default()
                    },
                    [data, idx],
                )
            }
            Op::KernelOp(kop) => match kop {
                KernelOp::Literal => self.ir.push_var(
                    ir::Var {
                        op: KernelOp::Literal,
                        ty: var.ty,
                        data: var.data.literal().unwrap(),
                        ..Default::default()
                    },
                    [],
                ),
                _ => {
                    let deps = trace
                        .deps(id)
                        .into_iter()
                        .map(|id| self.collect(trace, *id))
                        .collect::<Vec<_>>();
                    self.ir.push_var(
                        ir::Var {
                            op: kop,
                            ty: var.ty,
                            ..Default::default()
                        },
                        deps,
                    )
                }
            },
            _ => todo!(), // Op::Buffer => self.collect_data(trace, id),
        };
        id
    }
    pub fn collect_data(&mut self, trace: &trace::Trace, id: trace::VarId) -> ir::VarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = trace.var(id);

        match trace.var(id).op.resulting_op() {
            Op::Buffer => {
                let buffer_id = self.push_buffer(id);
                self.ir.push_var(
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
                let dim = trace.var(id).extent.texture_dim();
                self.ir.push_var(
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
                self.ir.push_var(
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
    pub fn push_buffer(&mut self, id: trace::VarId) -> usize {
        *self.id2buffer.entry(id).or_insert_with(|| {
            let buffer_id = self.buffers.len();
            self.buffers.push(id);
            buffer_id
        })
    }
    pub fn push_texture(&mut self, id: trace::VarId) -> usize {
        *self.id2texture.entry(id).or_insert_with(|| {
            let texture_id = self.textures.len();
            self.textures.push(id);
            texture_id
        })
    }
    pub fn push_accel(&mut self, id: trace::VarId) -> usize {
        *self.id2accel.entry(id).or_insert_with(|| {
            let accel_id = self.accels.len();
            self.accels.push(id);
            accel_id
        })
    }
}
