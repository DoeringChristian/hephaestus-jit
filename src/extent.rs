use slotmap::DefaultKey;

use crate::backend;
use crate::tr::VarId;

/// Represents a Variables Extent.
#[derive(Debug, Default, Clone, PartialEq)]
pub enum Extent {
    #[default]
    None, // TODO: actually use none...
    Size(usize),
    DynSize {
        capacity: usize,
        size_dep: VarId, // TODO: Need a better way to track size var
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
            (Extent::DynSize { .. }, Extent::Size(_)) => Some(std::cmp::Ordering::Less),
            (Extent::Size(_), Extent::DynSize { .. }) => Some(std::cmp::Ordering::Greater),
            (
                Extent::DynSize {
                    capacity: self_cap,
                    size_dep: self_size,
                },
                Extent::DynSize {
                    capacity: other_cap,
                    size_dep: other_size,
                },
            ) => Some(self_cap.cmp(other_cap).then(self_size.cmp(other_size))),
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
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Extent::DynSize { .. })
    }
    pub fn get_dynamic(&self) -> Option<VarId> {
        match self {
            Extent::DynSize { size_dep, .. } => Some(*size_dep),
            _ => None,
        }
    }
    pub fn accel_desc(&self) -> &backend::AccelDesc {
        match self {
            Extent::Accel(desc) => desc,
            _ => todo!(),
        }
    }
    pub fn resulting_extent(&self, other: &Self) -> Self {
        match (self, other) {
            (Extent::None, other) | (other, Extent::None) => other.clone(),
            (Extent::Size(a), Extent::Size(b)) => Extent::Size(*a.max(b)),
            (Extent::Size(a), Extent::DynSize { capacity, size_dep })
            | (Extent::DynSize { capacity, size_dep }, Extent::Size(a)) => Extent::DynSize {
                capacity: *a.max(capacity),
                size_dep: *size_dep,
            },
            (Extent::Size(size), _) | (_, Extent::Size(size)) => Extent::Size(*size),
            _ => todo!(),
        }
    }
    pub fn shape_and_channles(&self) -> ([usize; 3], usize) {
        match self {
            Extent::Texture { shape, channels } => (*shape, *channels),
            _ => todo!(),
        }
    }
    pub fn texture_dim(&self) -> usize {
        let (shape, _) = self.shape_and_channles();
        let dim = shape.iter().take_while(|d| **d > 0).count();
        dim
    }
}
