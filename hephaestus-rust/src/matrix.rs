use crate::{Vector3, Vector4};

#[derive(Clone, Debug)]
pub struct Matrix3<T: jit::AsVarType> {
    col0: Vector3<T>,
    col1: Vector3<T>,
    col2: Vector3<T>,
}

impl<T: jit::AsVarType> jit::Traverse for Matrix3<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [
            self.col0.traverse(vars),
            self.col1.traverse(vars),
            self.col2.traverse(vars),
        ];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::mat(&[self.col0.ravel(), self.col1.ravel(), self.col2.ravel()])
    }
}

impl<T: jit::AsVarType> jit::Construct for Matrix3<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &'static jit::Layout,
    ) -> Self {
        let mut layouts = layout.tuple_types().unwrap().into_iter();
        Self {
            col0: <Vector3<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            col1: <Vector3<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            col2: <Vector3<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
        }
    }
    fn unravel(var: impl AsRef<jit::VarRef>) -> Self {
        let var = var.as_ref();
        let ty = var.ty();
        assert!(
            matches!(ty, jit::vartype::VarType::Mat { cols, rows, ty: mat_ty } if *rows == 3 && *cols == 3 && &ty == mat_ty )
        );
        let mut iter = var.extract_all();
        Self {
            col0: <Vector3<T> as jit::Construct>::unravel(iter.next().unwrap()),
            col1: <Vector3<T> as jit::Construct>::unravel(iter.next().unwrap()),
            col2: <Vector3<T> as jit::Construct>::unravel(iter.next().unwrap()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Matrix4<T: jit::AsVarType> {
    col0: Vector4<T>,
    col1: Vector4<T>,
    col2: Vector4<T>,
    col3: Vector4<T>,
}

impl<T: jit::AsVarType> jit::Traverse for Matrix4<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [
            self.col0.traverse(vars),
            self.col1.traverse(vars),
            self.col2.traverse(vars),
            self.col3.traverse(vars),
        ];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::mat(&[
            self.col0.ravel(),
            self.col1.ravel(),
            self.col2.ravel(),
            self.col3.ravel(),
        ])
    }
}

impl<T: jit::AsVarType> jit::Construct for Matrix4<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &'static jit::Layout,
    ) -> Self {
        let mut layouts = layout.tuple_types().unwrap().into_iter();
        Self {
            col0: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            col1: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            col2: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            col3: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
        }
    }
    fn unravel(var: impl AsRef<jit::VarRef>) -> Self {
        let var = var.as_ref();
        let ty = var.ty();
        assert!(
            matches!(ty, jit::vartype::VarType::Mat { cols, rows, ty: mat_ty } if *rows == 4 && *cols == 4 && &ty == mat_ty )
        );
        let mut iter = var.extract_all();
        Self {
            col0: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
            col1: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
            col2: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
            col3: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
        }
    }
}

pub type Matrix3f = Matrix3<f32>;
pub type Matrix4f = Matrix4<f32>;

pub type Matrix3d = Matrix3<f64>;
pub type Matrix4d = Matrix4<f64>;

pub type Matrix3i = Matrix3<i32>;
pub type Matrix4i = Matrix4<i32>;

pub type Matrix3u = Matrix3<u32>;
pub type Matrix4u = Matrix4<u32>;

impl<T: jit::AsVarType> Matrix3<T> {
    pub fn from_cols(
        col0: impl AsRef<Vector3<T>>,
        col1: impl AsRef<Vector3<T>>,
        col2: impl AsRef<Vector3<T>>,
    ) -> Self {
        Self {
            col0: col0.as_ref().clone(),
            col1: col1.as_ref().clone(),
            col2: col2.as_ref().clone(),
        }
    }
}

impl<T: jit::AsVarType> Matrix4<T> {
    pub fn from_cols(
        col0: impl AsRef<Vector4<T>>,
        col1: impl AsRef<Vector4<T>>,
        col2: impl AsRef<Vector4<T>>,
        col3: impl AsRef<Vector4<T>>,
    ) -> Self {
        Self {
            col0: col0.as_ref().clone(),
            col1: col1.as_ref().clone(),
            col2: col2.as_ref().clone(),
            col3: col3.as_ref().clone(),
        }
    }
}

impl<T: jit::AsVarType> AsRef<Matrix3<T>> for Matrix3<T> {
    fn as_ref(&self) -> &Matrix3<T> {
        self
    }
}
impl<T: jit::AsVarType> AsRef<Matrix4<T>> for Matrix4<T> {
    fn as_ref(&self) -> &Matrix4<T> {
        self
    }
}
