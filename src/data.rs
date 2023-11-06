use crate::backend::Buffer;

///
/// A variable can hold data directly i.e. literals, buffers, textures or acceleration structures.
///
#[derive(Debug, Default, Clone)]
pub enum Data {
    #[default]
    None,
    Literal(u64),
    Buffer(Buffer),
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
        self.is_buffer()
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
}
