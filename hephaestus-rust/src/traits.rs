use super::var::Var;

pub trait Scatter {
    fn scatter(&self, dst: impl AsRef<Self>, index: impl AsRef<Var<u32>>);

    fn scatter_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    );
}

impl<T: jit::Traverse> Scatter for T {
    fn scatter(&self, dst: impl AsRef<Self>, index: impl AsRef<Var<u32>>) {
        let dst = dst.as_ref();
        let index = index.as_ref();

        let mut src_vars = vec![];
        let mut dst_vars = vec![];

        self.traverse(&mut src_vars);
        dst.traverse(&mut dst_vars);

        for (src, dst) in src_vars.into_iter().zip(dst_vars.into_iter()) {
            src.scatter(dst, index.clone());
        }
    }

    fn scatter_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    ) {
        let dst = dst.as_ref();
        let index = index.as_ref();
        let condition = condition.as_ref();

        let mut src_vars = vec![];
        let mut dst_vars = vec![];

        self.traverse(&mut src_vars);
        dst.traverse(&mut dst_vars);

        for (src, dst) in src_vars.into_iter().zip(dst_vars.into_iter()) {
            src.scatter_if(dst, index.clone(), condition.clone());
        }
        todo!()
    }
}

pub trait Gather {
    fn gather(&self, index: impl AsRef<Var<u32>>) -> Self;

    fn gather_if(&self, index: impl AsRef<Var<u32>>, condition: impl AsRef<Var<bool>>) -> Self;
}

impl<T: jit::Traverse + jit::Construct> Gather for T {
    fn gather(&self, index: impl AsRef<Var<u32>>) -> Self {
        let index = index.as_ref();

        let mut vars = vec![];

        let layout = self.traverse(&mut vars);

        for var in vars.iter_mut() {
            *var = var.gather(index.clone())
        }

        Self::construct(&mut vars.into_iter(), layout)
    }

    fn gather_if(&self, index: impl AsRef<Var<u32>>, condition: impl AsRef<Var<bool>>) -> Self {
        let index = index.as_ref();
        let condition = condition.as_ref();

        let mut vars = vec![];

        let layout = self.traverse(&mut vars);

        for var in vars.iter_mut() {
            *var = var.gather_if(index.clone(), condition.clone())
        }

        Self::construct(&mut vars.into_iter(), layout)
    }
}
