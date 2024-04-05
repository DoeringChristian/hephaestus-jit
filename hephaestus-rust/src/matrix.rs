use crate::{Vector3, Vector4};

#[derive(Clone, Debug)]
pub struct Matrix3<T: jit::AsVarType> {
    x_axis: Vector3<T>,
    y_axis: Vector3<T>,
    z_axis: Vector3<T>,
}

impl<T: jit::AsVarType> jit::Traverse for Matrix3<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [
            self.x_axis.traverse(vars),
            self.y_axis.traverse(vars),
            self.z_axis.traverse(vars),
        ];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::mat(&[
            self.x_axis.ravel(),
            self.y_axis.ravel(),
            self.z_axis.ravel(),
        ])
    }
}

impl<T: jit::AsVarType> jit::Construct for Matrix3<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &'static jit::Layout,
    ) -> Self {
        let mut layouts = layout.tuple_types().unwrap().into_iter();
        Self {
            x_axis: <Vector3<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            y_axis: <Vector3<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            z_axis: <Vector3<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
        }
    }
    fn unravel(var: impl Into<jit::VarRef>) -> Self {
        let var = var.into();
        let ty = var.ty();
        assert!(
            matches!(ty, jit::vartype::VarType::Mat { cols, rows, ty: mat_ty } if *rows == 3 && *cols == 3 && &ty == mat_ty )
        );
        let mut iter = var.extract_all();
        Self {
            x_axis: <Vector3<T> as jit::Construct>::unravel(iter.next().unwrap()),
            y_axis: <Vector3<T> as jit::Construct>::unravel(iter.next().unwrap()),
            z_axis: <Vector3<T> as jit::Construct>::unravel(iter.next().unwrap()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Matrix4<T: jit::AsVarType> {
    x_axis: Vector4<T>,
    y_axis: Vector4<T>,
    z_axis: Vector4<T>,
    w_axis: Vector4<T>,
}

impl<T: jit::AsVarType> jit::Traverse for Matrix4<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [
            self.x_axis.traverse(vars),
            self.y_axis.traverse(vars),
            self.z_axis.traverse(vars),
            self.w_axis.traverse(vars),
        ];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::mat(&[
            self.x_axis.ravel(),
            self.y_axis.ravel(),
            self.z_axis.ravel(),
            self.w_axis.ravel(),
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
            x_axis: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            y_axis: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            z_axis: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            w_axis: <Vector4<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
        }
    }
    fn unravel(var: impl Into<jit::VarRef>) -> Self {
        let var = var.into();
        let ty = var.ty();
        assert!(
            matches!(ty, jit::vartype::VarType::Mat { cols, rows, ty: mat_ty } if *rows == 4 && *cols == 4 && &ty == mat_ty )
        );
        let mut iter = var.extract_all();
        Self {
            x_axis: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
            y_axis: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
            z_axis: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
            w_axis: <Vector4<T> as jit::Construct>::unravel(iter.next().unwrap()),
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
        col0: impl Into<Vector3<T>>,
        col1: impl Into<Vector3<T>>,
        col2: impl Into<Vector3<T>>,
    ) -> Self {
        Self {
            x_axis: col0.into(),
            y_axis: col1.into(),
            z_axis: col2.into(),
        }
    }
}

impl<T: jit::AsVarType> Matrix4<T> {
    pub fn from_cols(
        col0: impl Into<Vector4<T>>,
        col1: impl Into<Vector4<T>>,
        col2: impl Into<Vector4<T>>,
        col3: impl Into<Vector4<T>>,
    ) -> Self {
        Self {
            x_axis: col0.into(),
            y_axis: col1.into(),
            z_axis: col2.into(),
            w_axis: col3.into(),
        }
    }
}

impl<T: jit::AsVarType> From<&Matrix3<T>> for Matrix3<T> {
    fn from(value: &Matrix3<T>) -> Self {
        value.clone()
    }
}
impl<T: jit::AsVarType> From<&Matrix4<T>> for Matrix4<T> {
    fn from(value: &Matrix4<T>) -> Self {
        value.clone()
    }
}

impl<T: jit::AsVarType> Matrix4<T> {
    pub fn mul_vec4(&self, rhs: impl Into<Vector4<T>>) -> Vector4<T> {
        todo!()
    }
}
