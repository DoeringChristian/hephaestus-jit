use crate::utils;
use half::f16;
use hephaestus_macros::AsVarType;
use once_cell::sync::Lazy;
use std::any::TypeId;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

// TODO: create a Type struct, wrapping &'static VarType

///
/// A hashmap, mapping a hash value to a leaked VarType reference.
///
static TYPE_CACHE: Lazy<Mutex<HashMap<u64, &'static VarType>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn from_ty<T: 'static>(f: impl Fn() -> VarType) -> &'static VarType {
    let ty_id = TypeId::of::<T>();

    let mut hasher = DefaultHasher::new();
    ty_id.hash(&mut hasher);
    0u32.hash(&mut hasher);
    let id = hasher.finish();

    TYPE_CACHE
        .lock()
        .unwrap()
        .entry(id)
        .or_insert_with(|| Box::leak(Box::new(f())))
}
pub fn composite(tys: &[&'static VarType]) -> &'static VarType {
    let mut hasher = DefaultHasher::new();
    tys.hash(&mut hasher);
    1u32.hash(&mut hasher);
    let id = hasher.finish();

    TYPE_CACHE
        .lock()
        .unwrap()
        .entry(id)
        .or_insert_with(|| Box::leak(Box::new(VarType::Struct { tys: tys.to_vec() })))
}
pub fn vector(ty: &'static VarType, num: usize) -> &'static VarType {
    let mut hasher = DefaultHasher::new();
    ty.hash(&mut hasher);
    num.hash(&mut hasher);
    2u32.hash(&mut hasher);
    let id = hasher.finish();

    TYPE_CACHE
        .lock()
        .unwrap()
        .entry(id)
        .or_insert_with(|| Box::leak(Box::new(VarType::Vec { ty, num })))
}
pub fn matrix(ty: &'static VarType, cols: usize, rows: usize) -> &'static VarType {
    let mut hasher = DefaultHasher::new();
    ty.hash(&mut hasher);
    cols.hash(&mut hasher);
    rows.hash(&mut hasher);
    3u32.hash(&mut hasher);
    let id = hasher.finish();

    TYPE_CACHE
        .lock()
        .unwrap()
        .entry(id)
        .or_insert_with(|| Box::leak(Box::new(VarType::Mat { ty, cols, rows })))
}
pub fn array(ty: &'static VarType, num: usize) -> &'static VarType {
    let mut hasher = DefaultHasher::new();
    ty.hash(&mut hasher);
    num.hash(&mut hasher);
    4u32.hash(&mut hasher);
    let id = hasher.finish();

    TYPE_CACHE
        .lock()
        .unwrap()
        .entry(id)
        .or_insert_with(|| Box::leak(Box::new(VarType::Array { ty, num })))
}
pub fn void() -> &'static VarType {
    from_ty::<()>(|| VarType::Void)
}

#[derive(Debug, PartialEq, Eq, Hash, Default)]
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
    F16,
    F32,
    F64,
    Vec {
        ty: &'static VarType,
        num: usize,
    },
    Array {
        ty: &'static VarType,
        num: usize,
    },
    Mat {
        ty: &'static VarType,
        rows: usize,
        cols: usize,
    },
    Struct {
        tys: Vec<&'static VarType>,
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
            VarType::F16 => 2,
            VarType::F32 => 4,
            VarType::F64 => 8,
            VarType::Vec { ty, num } => ty.size() * num,
            VarType::Struct { tys } => {
                let mut offset = 0;
                for i in 0..tys.len() - 1 {
                    offset += tys[i].size();
                    offset = utils::usize::align_up(offset, tys[i + 1].alignment());
                }
                return utils::usize::align_up(
                    offset + tys.last().unwrap().size(),
                    self.alignment(),
                );
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
                    offset = utils::usize::align_up(offset, tys[i + 1].alignment());
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
            VarType::F16 => 2,
            VarType::F32 => 4,
            VarType::F64 => 8,
            VarType::Vec { ty, num } => ty.alignment(),
            VarType::Struct { tys } => tys.iter().map(|ty| ty.alignment()).max().unwrap(),
            VarType::Mat { ty, cols, rows } => ty.size() * rows,
            VarType::Array { ty, num } => ty.alignment(),
        }
    }
    pub fn num_elements(&self) -> Option<usize> {
        match self {
            VarType::Vec { ty, num } => Some(*num),
            VarType::Array { ty, num } => Some(*num),
            VarType::Mat { ty, rows, cols } => Some(rows * cols),
            VarType::Struct { tys } => Some(tys.len()),
            _ => None,
        }
    }
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            VarType::I8
                | VarType::U8
                | VarType::I16
                | VarType::U16
                | VarType::I32
                | VarType::U32
                | VarType::I64
                | VarType::U64
        )
    }
    pub fn is_sint(&self) -> bool {
        matches!(
            self,
            VarType::I8 | VarType::I16 | VarType::I32 | VarType::I64
        )
    }
    pub fn is_uint(&self) -> bool {
        matches!(
            self,
            VarType::U8 | VarType::U16 | VarType::U32 | VarType::U64
        )
    }
    pub fn is_float(&self) -> bool {
        matches!(self, VarType::F16 | VarType::F32 | VarType::F64)
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, VarType::Bool)
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
    f16 => F16;
    f32 => F32;
    f64 => F64;
}

/// Instance Type used for Ray Tracing
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, AsVarType)]
pub struct Instance {
    pub transform: [f32; 12],
    pub geometry: u32,
}

/// Ray Intersection
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, AsVarType)]
pub struct Intersection {
    pub bx: f32,
    pub by: f32,
    pub instance_id: u32,
    pub primitive_idx: u32,
    pub valid: u32, // 0 if invalid, >0 if valid
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, AsVarType)]
pub struct Ray3f {
    pub o: mint::Point3<f32>,
    pub d: mint::Vector3<f32>,
    pub tmin: f32,
    pub tmax: f32,
}

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, AsVarType)]
pub struct MatMulConfig {
    pub M: u32,
    pub N: u32,
    pub K: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod, AsVarType)]
pub struct FusedMlpConfig {
    pub batch_size: u32,
}

impl<const N: usize, T: AsVarType + 'static> AsVarType for [T; N] {
    fn var_ty() -> &'static VarType {
        let ty = T::var_ty();
        array(ty, N)
    }
}

// MINT types
impl<T: AsVarType> AsVarType for mint::Vector2<T> {
    fn var_ty() -> &'static VarType {
        vector(T::var_ty(), 2)
    }
}
impl<T: AsVarType> AsVarType for mint::Vector3<T> {
    fn var_ty() -> &'static VarType {
        vector(T::var_ty(), 3)
    }
}
impl<T: AsVarType> AsVarType for mint::Vector4<T> {
    fn var_ty() -> &'static VarType {
        vector(T::var_ty(), 4)
    }
}
impl<T: AsVarType> AsVarType for mint::Point2<T> {
    fn var_ty() -> &'static VarType {
        vector(T::var_ty(), 2)
    }
}
impl<T: AsVarType> AsVarType for mint::Point3<T> {
    fn var_ty() -> &'static VarType {
        vector(T::var_ty(), 3)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix2<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 2, 2)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix2x3<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 3, 2)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix2x4<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 4, 2)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix3<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 3, 3)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix3x2<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 2, 3)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix3x4<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 4, 3)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix4<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 4, 4)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix4x2<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 2, 4)
    }
}
impl<T: AsVarType> AsVarType for mint::ColumnMatrix4x3<T> {
    fn var_ty() -> &'static VarType {
        matrix(T::var_ty(), 3, 4)
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

        let u8_ty = u8::var_ty();
        let u32_ty = u32::var_ty();
        let ty = from_ty::<Reference>(|| VarType::Struct {
            tys: vec![u8_ty, u32_ty],
        });

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

        let u8_ty = u8::var_ty();
        let u32_ty = u32::var_ty();
        let ty = from_ty::<Reference>(|| VarType::Struct {
            tys: vec![u32_ty, u8_ty],
        });

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

        let u8_ty = u8::var_ty();
        let u16_ty = u16::var_ty();
        let u32_ty = u32::var_ty();
        let u64_ty = u64::var_ty();
        let ty = from_ty::<Reference>(|| VarType::Struct {
            tys: vec![u8_ty, u16_ty, u32_ty, u64_ty],
        });

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
