use crate::backend::{Accel, Buffer, Device, Texture};
pub use crate::backend::{AccelDesc, BufferDesc, TextureDesc};
use crate::utils;
use crate::vartype::VarType;
// TODO: maybe move to backend?
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceDesc {
    BufferDesc(BufferDesc),
    TextureDesc(TextureDesc),
    AccelDesc(AccelDesc),
}
impl ResourceDesc {
    pub fn round_up(&self) -> Self {
        match self {
            ResourceDesc::BufferDesc(desc) => ResourceDesc::BufferDesc(BufferDesc {
                size: utils::u64::round_pow2(desc.size as _) as usize,
                ty: desc.ty,
            }),
            ResourceDesc::TextureDesc(desc) => ResourceDesc::TextureDesc(desc.clone()),
            ResourceDesc::AccelDesc(desc) => todo!(),
        }
    }
}

///
/// A variable can hold data directly i.e. literals, buffers, textures or acceleration structures.
///
#[derive(Debug, Default, Clone)]
pub enum Resource {
    #[default]
    None,
    Literal(u64),
    Buffer(Buffer),
    Texture(Texture),
    Accel(Accel),
}
impl Resource {
    pub fn create(device: &Device, desc: &ResourceDesc) -> Self {
        match desc {
            ResourceDesc::BufferDesc(desc) => {
                Resource::Buffer(device.create_buffer(desc.size * desc.ty.size()).unwrap())
            }
            ResourceDesc::TextureDesc(desc) => {
                Resource::Texture(device.create_texture(desc).unwrap())
            }
            ResourceDesc::AccelDesc(desc) => Resource::Accel(device.create_accel(desc).unwrap()),
        }
    }
    pub fn match_and_get(&self, desc: &ResourceDesc) -> Option<Self> {
        // TODO: match sizes as well.
        match (self, desc) {
            (Resource::Buffer(_), ResourceDesc::BufferDesc(_)) => Some(self.clone()),
            (Resource::Texture(_), ResourceDesc::TextureDesc(_)) => Some(self.clone()),
            (Resource::Accel(_), ResourceDesc::AccelDesc(_)) => Some(self.clone()),
            _ => None,
        }
    }
    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            _ => false,
        }
    }
    pub fn is_literal(&self) -> bool {
        match self {
            Self::Literal(_) => true,
            _ => false,
        }
    }
    pub fn is_buffer(&self) -> bool {
        match self {
            Self::Buffer(_) => true,
            _ => false,
        }
    }
    pub fn is_storage(&self) -> bool {
        self.is_buffer() || self.is_texture()
    }
    pub fn literal(&self) -> Option<u64> {
        match self {
            Self::Literal(lit) => Some(*lit),
            _ => None,
        }
    }
    pub fn buffer(&self) -> Option<&Buffer> {
        match self {
            Self::Buffer(buf) => Some(buf),
            _ => None,
        }
    }

    pub fn is_texture(&self) -> bool {
        match self {
            Self::Texture(_) => true,
            _ => false,
        }
    }
    pub fn texture(&self) -> Option<&Texture> {
        match self {
            Self::Texture(tex) => Some(tex),
            _ => None,
        }
    }
    pub fn accel(&self) -> Option<&Accel> {
        match self {
            Self::Accel(accel) => Some(accel),
            _ => None,
        }
    }
}
