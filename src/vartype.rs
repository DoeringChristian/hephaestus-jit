use std::sync::Arc;

use lazy_static::lazy_static;

const fn align_up(v: usize, base: usize) -> usize {
    ((v + base - 1) / base) * base
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum VarType {
    // Primitive Types (might move out)
    #[default]
    Void,
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Vec {
        ty: Box<VarType>,
        num: usize,
    },
    Array {
        ty: Box<VarType>,
        num: usize,
    },
    Mat {
        ty: Box<VarType>,
        rows: usize,
        cols: usize,
    },
    Struct {
        tys: Vec<VarType>,
    },
}
impl VarType {
    // TODO: Check that alignment calculations are correct
    pub fn size(&self) -> usize {
        match self {
            VarType::Void => 0,
            VarType::Bool => 1,
            VarType::I8 => 1,
            VarType::U8 => 1,
            VarType::I16 => 2,
            VarType::U16 => 2,
            VarType::I32 => 4,
            VarType::U32 => 4,
            VarType::I64 => 8,
            VarType::U64 => 8,
            VarType::F32 => 4,
            VarType::F64 => 8,
            VarType::Vec { ty, num } => ty.size() * num,
            VarType::Struct { tys } => {
                let mut offset = 0;
                for i in 0..tys.len() - 1 {
                    offset += tys[i].size();
                    offset = align_up(offset, tys[i + 1].alignment());
                }
                return align_up(offset + tys.last().unwrap().size(), self.alignment());
            }
            VarType::Mat { ty, cols, rows } => ty.size() * cols * rows,
            VarType::Array { ty, num } => ty.size() * num,
        }
    }
    pub fn offset(&self, elem: usize) -> usize {
        match self {
            VarType::Struct { tys } => {
                let mut offset = 0;
                for i in 0..elem {
                    offset += tys[i].size();
                    offset = align_up(offset, tys[i + 1].alignment());
                }
                offset
            }
            _ => todo!(),
        }
    }
    pub fn alignment(&self) -> usize {
        match self {
            VarType::Void => 0,
            VarType::Bool => 1,
            VarType::I8 => 1,
            VarType::U8 => 1,
            VarType::I16 => 2,
            VarType::U16 => 2,
            VarType::I32 => 4,
            VarType::U32 => 4,
            VarType::I64 => 8,
            VarType::U64 => 8,
            VarType::F32 => 4,
            VarType::F64 => 8,
            VarType::Vec { ty, num } => ty.alignment(),
            VarType::Struct { tys } => tys.iter().map(|ty| ty.alignment()).max().unwrap(),
            VarType::Mat { ty, cols, rows } => ty.size() * rows,
            VarType::Array { ty, num } => ty.alignment(),
        }
    }
    pub fn is_int(&self) -> bool {
        match self {
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
    pub fn is_sint(&self) -> bool {
        match self {
            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64 => true,
            _ => false,
        }
    }
    pub fn is_uint(&self) -> bool {
        match self {
            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64 => true,
            _ => false,
        }
    }
    pub fn is_float(&self) -> bool {
        match self {
            VarType::F32 | VarType::F64 => true,
            _ => false,
        }
    }
    pub fn is_bool(&self) -> bool {
        match self {
            VarType::Bool => true,
            _ => false,
        }
    }
}

pub trait AsVarType: Copy {
    fn var_ty() -> &'static VarType;
}
macro_rules! as_var_type {
    {$($src:ident => $dst:ident;)*} => {
        $(as_var_type!($src => $dst);)*
    };
    ($src:ident => $dst:ident) => {
        impl AsVarType for $src{
            fn var_ty() -> &'static VarType{
                lazy_static::lazy_static!{
                    static ref TY: VarType = VarType::$dst;
                }
                &TY
            }
        }
    };
}
as_var_type! {
    bool => Bool;
    i8 => I8;
    u8 => U8;
    i16 => I16;
    u16 => U16;
    i32 => I32;
    u32 => U32;
    i64 => I64;
    u64 => U64;
    f32 => F32;
    f64 => F64;
}

/// Instance Type used for Ray Tracing
#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Instance {
    pub transform: [f32; 12],
    pub geometry: u32,
}

impl AsVarType for Instance {
    fn var_ty() -> &'static VarType {
        lazy_static! {
            static ref TY: VarType = VarType::Struct {
                tys: vec![
                    VarType::Array {
                        ty: Box::new(VarType::F32),
                        num: 12,
                    },
                    VarType::U32,
                ],
            };
        };
        &TY
    }
}

/// Ray Intersection
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Intersection {
    pub barycentrics: [f32; 2],
    pub instance_id: u32,
    pub primitive_idx: u32,
    pub valid: u32, // 0 if invalid, >0 if valid
}

impl AsVarType for Intersection {
    fn var_ty() -> &'static VarType {
        lazy_static! {
            static ref TY: VarType = VarType::Struct {
                tys: vec![
                    VarType::Vec {
                        ty: Box::new(VarType::F32),
                        num: 2
                    },
                    VarType::U32,
                    VarType::U32,
                    VarType::U32,
                ],
            };
        };
        &TY
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn u8u32() {
        #[derive(Default)]
        #[repr(C)]
        struct Reference {
            a: u8,
            b: u32,
        }

        let ty = VarType::Struct {
            tys: vec![VarType::U8, VarType::U32],
        };

        assert_eq!(ty.offset(0), bytemuck::offset_of!(Reference, a));
        assert_eq!(ty.offset(1), bytemuck::offset_of!(Reference, b));
        assert_eq!(ty.size(), std::mem::size_of::<Reference>());
    }
    #[test]
    fn u32u8() {
        #[derive(Default)]
        #[repr(C)]
        struct Reference {
            a: u32,
            b: u8,
        }

        let ty = VarType::Struct {
            tys: vec![VarType::U32, VarType::U8],
        };

        assert_eq!(ty.offset(0), bytemuck::offset_of!(Reference, a));
        assert_eq!(ty.offset(1), bytemuck::offset_of!(Reference, b));
        assert_eq!(ty.size(), std::mem::size_of::<Reference>());
    }
    #[test]
    fn u8u16u32u64() {
        #[derive(Default)]
        #[repr(C)]
        struct Reference {
            a: u8,
            b: u16,
            c: u32,
            d: u64,
        }

        let ty = VarType::Struct {
            tys: vec![VarType::U8, VarType::U16, VarType::U32, VarType::U64],
        };

        assert_eq!(ty.offset(0), bytemuck::offset_of!(Reference, a));
        assert_eq!(ty.offset(1), bytemuck::offset_of!(Reference, b));
        assert_eq!(ty.offset(2), bytemuck::offset_of!(Reference, c));
        assert_eq!(ty.offset(3), bytemuck::offset_of!(Reference, d));
        assert_eq!(ty.size(), std::mem::size_of::<Reference>());
    }
    #[test]
    fn instance() {
        let ty = Instance::var_ty();

        assert_eq!(ty.offset(0), bytemuck::offset_of!(Instance, transform));
        assert_eq!(ty.offset(1), bytemuck::offset_of!(Instance, geometry));
        assert_eq!(ty.size(), std::mem::size_of::<Instance>());
    }
}
