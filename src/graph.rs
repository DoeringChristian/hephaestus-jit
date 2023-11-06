#[derive(Debug, Default, Clone)]
pub struct Node {
    pass: Pass,
    deps: (usize, usize),
}

#[derive(Default, Debug, Clone)]
pub struct Graph {
    passes: Vec<Node>,
    deps: Vec<PassId>,
}

impl Graph {
    pub fn push_pass(&mut self, pass: Pass, deps: impl IntoIterator<Item = PassId>) -> PassId {
        let id = PassId(self.passes.len());

        let start = self.deps.len();
        self.deps.extend(deps);
        let stop = self.deps.len();

        self.passes.push(Node {
            pass,
            deps: (start, stop),
        });
        id
    }
    pub fn pass(&self, id: PassId) -> &Pass {
        &self.passes[id.0].pass
    }
    pub fn pass_mut(&mut self, id: PassId) -> &mut Pass {
        &mut self.passes[id.0].pass
    }
    pub fn deps(&self, id: PassId) -> &[PassId] {
        let (start, stop) = self.passes[id.0].deps;
        &self.deps[start..stop]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassId(usize);

#[derive(Default, Debug, Clone)]
pub enum Pass {
    #[default]
    None,
}
