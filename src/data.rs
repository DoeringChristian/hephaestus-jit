use crate::backend::{Buffer, Texture};

///
/// A variable can hold data directly i.e. literals, buffers, textures or acceleration structures.
///
#[derive(Debug, Default)]
pub enum Data {
    #[default]
    None,
    Literal(u64),
    Buffer(Buffer),
    Texture(Texture),
}
impl Data {
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
}
