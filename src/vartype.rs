use std::sync::Arc;

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
        }
    }
    pub fn offset(&self, elem: usize) -> usize {
        match self {
            VarType::Struct { tys } => {
                let mut offset = 0;
                if elem == 0 {
                    return 0;
                }
                for i in 0..elem - 1 {
                    offset += tys[i].size();
                    offset = align_up(offset, tys[i + 1].alignment());
                }
                offset
            }
            _ => todo!(),
        }
    }
    // TODO: Fix allignment
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
            VarType::Vec { ty, num } => ty.size() * num,
            VarType::Struct { tys } => tys.iter().map(|ty| ty.alignment()).max().unwrap(),
        }
    }
}

pub trait AsVarType {
    fn var_ty() -> VarType;
}
macro_rules! as_var_type {
    {$($src:ident => $dst:ident;)*} => {
        $(as_var_type!($src => $dst);)*
    };
    ($src:ident => $dst:ident) => {
        impl AsVarType for $src{
            fn var_ty() -> VarType{
                VarType::$dst
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
