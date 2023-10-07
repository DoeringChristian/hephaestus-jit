#[derive(Clone, Copy)]
pub struct ParamLayout {
    pub size: usize,
    pub arrays: usize,
    pub n_params: usize,
}

impl ParamLayout {
    pub fn array_offset(&self, i: usize) -> usize {
        let idx = self.arrays + i;
        assert!(idx < self.n_params);
        idx * 8
    }
}
