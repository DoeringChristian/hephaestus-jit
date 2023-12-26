use crate::backend::{Accel, AccelDesc, Buffer, Device, Texture};
use crate::vartype::VarType;
// TODO: maybe move to backend?

#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub size: usize,
    pub ty: VarType,
}
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub shape: [usize; 3],
    pub channels: usize,
}
#[derive(Debug)]
pub enum ResourceDesc {
    BufferDesc(BufferDesc),
    TextureDesc(TextureDesc),
    AccelDesc(AccelDesc),
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
                Resource::Texture(device.create_texture(desc.shape, desc.channels).unwrap())
            }
            ResourceDesc::AccelDesc(desc) => Resource::Accel(device.create_accel(desc).unwrap()),
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
