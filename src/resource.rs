use crate::backend::{Accel, Array, Device, Texture};
pub use crate::backend::{AccelDesc, ArrayDesc, TextureDesc};
use crate::vartype::VarType;
// TODO: maybe move to backend?
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum ResourceDesc {
    ArrayDesc(ArrayDesc),
    TextureDesc(TextureDesc),
    AccelDesc(AccelDesc),
}

///
/// A variable can hold data directly i.e. literals, buffers, textures or acceleration structures.
///
#[derive(Debug, Clone)]
pub enum Resource {
    Array(Array),
    Texture(Texture),
    Accel(Accel),
}
impl Resource {
    pub fn create(device: &Device, desc: &ResourceDesc) -> Self {
        match desc {
            ResourceDesc::ArrayDesc(desc) => {
                Resource::Array(device.create_buffer(desc.size * desc.ty.size()).unwrap())
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
            (Resource::Array(_), ResourceDesc::ArrayDesc(_)) => Some(self.clone()),
            (Resource::Texture(_), ResourceDesc::TextureDesc(_)) => Some(self.clone()),
            (Resource::Accel(_), ResourceDesc::AccelDesc(_)) => Some(self.clone()),
            _ => None,
        }
    }
    pub fn is_buffer(&self) -> bool {
        match self {
            Self::Array(_) => true,
            _ => false,
        }
    }
    pub fn is_storage(&self) -> bool {
        self.is_buffer() || self.is_texture()
    }
    pub fn buffer(&self) -> Option<&Array> {
        match self {
            Self::Array(buf) => Some(buf),
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
