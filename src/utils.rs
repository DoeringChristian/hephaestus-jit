pub mod usize {
    use std::ops::Range;
    use std::ops::RangeBounds;

    pub fn limit_range_bounds(
        bounds: impl RangeBounds<usize>,
        limit: Range<usize>,
    ) -> Range<usize> {
        let start = match bounds.start_bound() {
            std::ops::Bound::Included(i) => usize::max(*i, limit.start),
            std::ops::Bound::Excluded(i) => usize::max(*i + 1, limit.start),
            std::ops::Bound::Unbounded => limit.start,
        };
        let end = match bounds.end_bound() {
            std::ops::Bound::Included(i) => usize::min(*i + 1, limit.end),
            std::ops::Bound::Excluded(i) => usize::min(*i, limit.end),
            std::ops::Bound::Unbounded => limit.end,
        };
        start..end
    }
}
