use std::hash::Hash;

use crate::var::{Add, Mul};
use crate::Scatter;

use super::var::Var;

#[derive(Clone, Debug)]
pub struct Vector2<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
}

impl<T: jit::AsVarType> From<&Vector2<T>> for Vector2<T> {
    fn from(value: &Vector2<T>) -> Self {
        value.clone()
    }
}

impl<T: jit::AsVarType> Hash for Vector2<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
    }
}

impl<T: jit::AsVarType> jit::Traverse for Vector2<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [self.x.traverse(vars), self.y.traverse(vars)];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::vec(&[self.x.0.clone(), self.y.0.clone()])
    }
}

impl<T: jit::AsVarType> jit::Construct for Vector2<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &'static jit::Layout,
    ) -> Self {
        let mut layouts = layout.tuple_types().unwrap().into_iter();
        Self {
            x: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            y: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
        }
    }
    fn unravel(var: impl Into<jit::VarRef>) -> Self {
        let var = var.into();
        let ty = var.ty();
        assert!(
            matches!(ty, jit::vartype::VarType::Vec { num, ty: vec_ty } if *num == 2 && &ty == vec_ty )
        );
        let mut iter = var.extract_all();
        Self {
            x: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
            y: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Vector3<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
    z: Var<T>,
}

impl<T: jit::AsVarType> From<&Vector3<T>> for Vector3<T> {
    fn from(value: &Vector3<T>) -> Self {
        value.clone()
    }
}

impl<T: jit::AsVarType> Hash for Vector3<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
    }
}

impl<T: jit::AsVarType> jit::Traverse for Vector3<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [
            self.x.traverse(vars),
            self.y.traverse(vars),
            self.z.traverse(vars),
        ];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::vec(&[self.x.0.clone(), self.y.0.clone(), self.z.0.clone()])
    }
}

impl<T: jit::AsVarType> jit::Construct for Vector3<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &'static jit::Layout,
    ) -> Self {
        let mut layouts = layout.tuple_types().unwrap().into_iter();
        Self {
            x: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            y: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            z: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
        }
    }
    fn unravel(var: impl Into<jit::VarRef>) -> Self {
        let var = var.into();
        let ty = var.ty();
        assert!(
            matches!(ty, jit::vartype::VarType::Vec { num, ty: vec_ty } if *num == 3 && &ty == vec_ty )
        );
        let mut iter = var.extract_all();
        Self {
            x: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
            y: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
            z: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Vector4<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
    z: Var<T>,
    w: Var<T>,
}

impl<T: jit::AsVarType> From<&Vector4<T>> for Vector4<T> {
    fn from(value: &Vector4<T>) -> Self {
        value.clone()
    }
}

impl<T: jit::AsVarType> Hash for Vector4<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
        self.w.hash(state);
    }
}

impl<T: jit::AsVarType> jit::Traverse for Vector4<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [
            self.x.traverse(vars),
            self.y.traverse(vars),
            self.z.traverse(vars),
            self.w.traverse(vars),
        ];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::vec(&[
            self.x.0.clone(),
            self.y.0.clone(),
            self.z.0.clone(),
            self.w.0.clone(),
        ])
    }
}

impl<T: jit::AsVarType> jit::Construct for Vector4<T> {
    fn construct(
        vars: &mut impl Iterator<Item = jit::VarRef>,
        layout: &'static jit::Layout,
    ) -> Self {
        let mut layouts = layout.tuple_types().unwrap().into_iter();
        Self {
            x: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            y: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            z: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
            w: <Var<T> as jit::Construct>::construct(vars, layouts.next().unwrap()),
        }
    }
    fn unravel(var: impl Into<jit::VarRef>) -> Self {
        let var = var.into();
        let ty = var.ty();
        assert!(
            matches!(ty, jit::vartype::VarType::Vec { num, ty: vec_ty } if *num == 3 && &ty == vec_ty )
        );
        let mut iter = var.extract_all();
        Self {
            x: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
            y: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
            z: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
            w: <Var<T> as jit::Construct>::unravel(iter.next().unwrap()),
        }
    }
}

pub type Vector2f = Var<Vector2<f32>>;
pub type Vector3f = Vector3<f32>;
pub type Vector4f = Vector4<f32>;

pub type Vector2d = Vector2<f64>;
pub type Vector3d = Vector3<f64>;
pub type Vector4d = Vector4<f64>;

pub type Vector2i = Vector2<i32>;
pub type Vector3i = Vector3<i32>;
pub type Vector4i = Vector4<i32>;

pub type Vector2u = Vector2<u32>;
pub type Vector3u = Vector3<u32>;
pub type Vector4u = Vector4<u32>;

pub fn vec2<T: jit::AsVarType>(x: impl Into<jit::VarRef>, y: impl Into<jit::VarRef>) -> Vector2<T> {
    Vector2::<T> {
        x: x.into().into(),
        y: y.into().into(),
    }
}
pub fn vec3<T: jit::AsVarType>(
    x: impl Into<jit::VarRef>,
    y: impl Into<jit::VarRef>,
    z: impl Into<jit::VarRef>,
) -> Vector3<T> {
    Vector3::<T> {
        x: x.into().into(),
        y: y.into().into(),
        z: z.into().into(),
    }
}
pub fn vec4<T: jit::AsVarType>(
    x: impl Into<jit::VarRef>,
    y: impl Into<jit::VarRef>,
    z: impl Into<jit::VarRef>,
    w: impl Into<jit::VarRef>,
) -> Vector4<T> {
    Vector4::<T> {
        x: x.into().into(),
        y: y.into().into(),
        z: z.into().into(),
        w: w.into().into(),
    }
}

impl<T: jit::AsVarType> From<jit::VarRef> for Vector2<T> {
    fn from(value: jit::VarRef) -> Self {
        Self {
            x: value.extract(0).into(),
            y: value.extract(1).into(),
        }
    }
}
impl<T: jit::AsVarType> From<jit::VarRef> for Vector3<T> {
    fn from(value: jit::VarRef) -> Self {
        Self {
            x: value.extract(0).into(),
            y: value.extract(1).into(),
            z: value.extract(2).into(),
        }
    }
}
impl<T: jit::AsVarType> From<jit::VarRef> for Vector4<T> {
    fn from(value: jit::VarRef) -> Self {
        Self {
            x: value.extract(0).into(),
            y: value.extract(1).into(),
            z: value.extract(2).into(),
            w: value.extract(3).into(),
        }
    }
}

impl<T: jit::AsVarType> Vector2<T> {
    pub fn ravel(&self) -> jit::VarRef {
        jit::vec(&[(&self.x).into(), (&self.y).into()])
    }
}
impl<T: jit::AsVarType> Vector3<T> {
    pub fn ravel(&self) -> jit::VarRef {
        jit::vec(&[(&self.x).into(), (&self.y).into(), (&self.z).into()])
    }
}
impl<T: jit::AsVarType> Vector4<T> {
    pub fn ravel(&self) -> jit::VarRef {
        jit::vec(&[
            (&self.x).into(),
            (&self.y).into(),
            (&self.z).into(),
            (&self.w).into(),
        ])
    }
}

impl<T: jit::AsVarType> Vector2<T>
where
    Vector2<T>: Scatter,
{
    pub fn scatter(&self, dst: impl Into<Self>, index: impl Into<Var<u32>>) {
        Scatter::scatter(self, dst, index)
    }
    pub fn scatter_if(
        &self,
        dst: impl Into<Self>,
        index: impl Into<Var<u32>>,
        condition: impl Into<Var<bool>>,
    ) {
        Scatter::scatter_if(self, dst, index, condition)
    }
}

impl<T: jit::AsVarType> Vector3<T>
where
    Vector3<T>: Scatter,
{
    pub fn scatter(&self, dst: impl Into<Self>, index: impl Into<Var<u32>>) {
        Scatter::scatter(self, dst, index)
    }
    pub fn scatter_if(
        &self,
        dst: impl Into<Self>,
        index: impl Into<Var<u32>>,
        condition: impl Into<Var<bool>>,
    ) {
        Scatter::scatter_if(self, dst, index, condition)
    }
}

impl<T: jit::AsVarType> Vector2<T>
where
    Var<T>: Add + Mul,
{
    pub fn dot(&self, rhs: impl AsRef<Self>) -> Var<T> {
        let rhs = rhs.as_ref();
        &self.x * &rhs.x + &self.y * &rhs.y
    }
}

impl<T: jit::AsVarType> std::ops::Mul<Var<T>> for Vector2<T> {
    type Output = Vector2<T>;

    fn mul(self, rhs: Var<T>) -> Self::Output {
        todo!()
    }
}

impl<T: jit::AsVarType> Vector3<T>
where
    Var<T>: Add + Mul,
{
    pub fn dot(&self, rhs: impl AsRef<Self>) -> Var<T> {
        let rhs = rhs.as_ref();
        &self.x * &rhs.x + &self.y * &rhs.y + &self.z * &rhs.z
    }
}

impl<T: jit::AsVarType> Vector4<T>
where
    Var<T>: Add + Mul,
{
    pub fn dot(&self, rhs: impl AsRef<Self>) -> Var<T> {
        let rhs = rhs.as_ref();
        &self.x * &rhs.x + &self.y * &rhs.y + &self.z * &rhs.z + &self.w * &rhs.w
    }
}
