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
    pub const fn align_up(val: usize, base: usize) -> usize {
        div_round_up(val, base) * base
    }
    pub const fn div_round_up(val: usize, divisor: usize) -> usize {
        (val + divisor - 1) / divisor
    }
}
pub mod u64 {
    pub fn round_pow2(x: u64) -> u64 {
        let x = x - 1;
        let x = x | x.overflowing_shr(1).0;
        let x = x | x.overflowing_shr(2).0;
        let x = x | x.overflowing_shr(4).0;
        let x = x | x.overflowing_shr(8).0;
        let x = x | x.overflowing_shr(16).0;
        let x = x | x.overflowing_shr(32).0;
        x + 1
    }
}
pub mod u32 {
    pub fn round_pow2(x: u32) -> u32 {
        let mut x = x;
        x -= 1;
        x |= x.overflowing_shr(1).0;
        x |= x.overflowing_shr(2).0;
        x |= x.overflowing_shr(4).0;
        x |= x.overflowing_shr(8).0;
        x |= x.overflowing_shr(16).0;
        x |= x.overflowing_shr(32).0;
        return x + 1;
    }
}
