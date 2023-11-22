use crate::backend;

/// Represents a variables extent
#[derive(Debug, Default, Clone, PartialEq)]
pub enum Extent {
    #[default]
    None,
    Size(usize),
    DynSize {
        capacity: usize,
        size_dep: usize,
    },
    Texture {
        shape: [usize; 3],
        channels: usize,
    },
    Accel(backend::AccelDesc),
}
impl PartialOrd for Extent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Extent::Size(a), Extent::Size(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
}
impl Extent {
    pub fn capacity(&self) -> usize {
        match self {
            Extent::Size(size) => *size,
            Extent::DynSize { capacity, .. } => *capacity,
            _ => todo!(),
        }
    }
    pub fn size(&self) -> usize {
        match self {
            Extent::Size(size) => *size,
            _ => todo!(),
        }
    }
}
