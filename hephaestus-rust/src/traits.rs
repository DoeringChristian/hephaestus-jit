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

pub trait Select {
    fn select(&self, condition: impl AsRef<Var<bool>>, false_val: impl AsRef<Self>) -> Self;
}

impl<T: jit::Traverse + jit::Construct> Select for T {
    fn select(&self, condition: impl AsRef<Var<bool>>, false_val: impl AsRef<Self>) -> Self {
        let condition = condition.as_ref();
        let false_val = false_val.as_ref();

        let mut true_vars = vec![];
        let mut false_vars = vec![];

        let true_layout = self.traverse(&mut true_vars);
        let false_layout = false_val.traverse(&mut false_vars);

        assert_eq!(true_layout, false_layout);

        for (true_var, false_var) in true_vars.iter_mut().zip(false_vars.iter()) {
            *true_var = true_var.select(condition, false_var);
        }

        Self::construct(&mut true_vars.into_iter(), true_layout)
    }
}
