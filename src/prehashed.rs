use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;

#[derive(Clone, Copy)]
pub struct Prehashed<T: ?Sized> {
    hash: u64,

    item: T,
}

impl<T: Hash + 'static> Prehashed<T> {
    pub fn new(item: T) -> Self {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        Self { hash, item }
    }

    pub fn into_inner(self) -> T {
        self.item
    }
}

impl<T> Hash for Prehashed<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

impl<T> Deref for Prehashed<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<T: Hash + 'static> From<T> for Prehashed<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: Debug + ?Sized> Debug for Prehashed<T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        self.item.fmt(f)
    }
}

impl<T: Default + Hash + 'static> Default for Prehashed<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: ?Sized> Eq for Prehashed<T> {}

impl<T: ?Sized> PartialEq for Prehashed<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl<T: Ord + ?Sized> Ord for Prehashed<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.item.cmp(&other.item)
    }
}

impl<T: PartialOrd + ?Sized> PartialOrd for Prehashed<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.item.partial_cmp(&other.item)
    }
}
