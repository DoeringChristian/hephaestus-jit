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
}
