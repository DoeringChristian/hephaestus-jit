use itertools::Itertools;
use std::iter::Peekable;

pub trait RangeGroupBy<I: Iterator> {
    fn range_group_by<P: FnMut(&I::Item, &I::Item) -> bool>(
        self,
        predicate: P,
    ) -> IterRangeGroupBy<I, P>;
}

impl<I: Iterator> RangeGroupBy<I> for I {
    fn range_group_by<P: FnMut(&<I as Iterator>::Item, &<I as Iterator>::Item) -> bool>(
        self,
        predicate: P,
    ) -> IterRangeGroupBy<I, P> {
        IterRangeGroupBy {
            iterator: self.peekable(),
            predicate,
            count: 0,
        }
    }
}

pub struct IterRangeGroupBy<I: Iterator, P> {
    iterator: Peekable<I>,
    predicate: P,
    count: usize,
}

impl<T, I: Iterator<Item = T>, P: FnMut(&T, &T) -> bool> Iterator for IterRangeGroupBy<I, P> {
    type Item = std::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.count;
        let mut count = self.count + 1;
        let mut current = self.iterator.next()?;
        loop {
            if let Some(next) = self.iterator.peek() {
                if (self.predicate)(&current, next) {
                    current = self.iterator.next().unwrap();
                    count += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        self.count = count;
        Some(start..count)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn range_group_by_01() {
        let v = [0, 0, 1, 1, 0, 1]
            .into_iter()
            .range_group_by(|&a, &b| a == b)
            .collect::<Vec<_>>();
        assert_eq!(v, vec![0..2, 2..4, 4..5, 5..6]);
    }
    #[test]
    fn range_group_by_02() {
        let v = [0]
            .into_iter()
            .range_group_by(|&a, &b| a == b)
            .collect::<Vec<_>>();
        assert_eq!(v, vec![0..1]);
    }
    #[test]
    fn range_group_by_03() {
        let v = [0]
            .into_iter()
            .range_group_by(|&a, &b| a == b)
            .collect::<Vec<_>>();
        assert_eq!(v, vec![0..1]);
    }
}
