use crate::Scatter;

use super::var::Var;

#[derive(jit::Traverse, jit::Construct, Clone, Debug)]
pub struct Vector2<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
}
#[derive(jit::Traverse, jit::Construct, Clone, Debug)]
pub struct Vector3<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
    z: Var<T>,
}
#[derive(jit::Traverse, jit::Construct, Clone, Debug)]
pub struct Vector4<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
    z: Var<T>,
    w: Var<T>,
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

pub fn vec2<T: jit::AsVarType>(
    x: impl AsRef<jit::VarRef>,
    y: impl AsRef<jit::VarRef>,
) -> Vector2<T> {
    Vector2::<T> {
        x: x.as_ref().into(),
        y: y.as_ref().into(),
    }
}
pub fn vec3<T: jit::AsVarType>(
    x: impl AsRef<jit::VarRef>,
    y: impl AsRef<jit::VarRef>,
    z: impl AsRef<jit::VarRef>,
) -> Vector3<T> {
    Vector3::<T> {
        x: x.as_ref().into(),
        y: y.as_ref().into(),
        z: z.as_ref().into(),
    }
}
pub fn vec4<T: jit::AsVarType>(
    x: impl AsRef<jit::VarRef>,
    y: impl AsRef<jit::VarRef>,
    z: impl AsRef<jit::VarRef>,
    w: impl AsRef<jit::VarRef>,
) -> Vector4<T> {
    Vector4::<T> {
        x: x.as_ref().into(),
        y: y.as_ref().into(),
        z: z.as_ref().into(),
        w: w.as_ref().into(),
    }
}

impl<T: jit::AsVarType> AsRef<Vector2<T>> for Vector2<T> {
    fn as_ref(&self) -> &Vector2<T> {
        self
    }
}
impl<T: jit::AsVarType> AsRef<Vector3<T>> for Vector3<T> {
    fn as_ref(&self) -> &Vector3<T> {
        self
    }
}
impl<T: jit::AsVarType> AsRef<Vector4<T>> for Vector4<T> {
    fn as_ref(&self) -> &Vector4<T> {
        self
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
        jit::vec(&[
            AsRef::<jit::VarRef>::as_ref(&self.x).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.y).clone(),
        ])
    }
}
impl<T: jit::AsVarType> Vector3<T> {
    pub fn ravel(&self) -> jit::VarRef {
        jit::vec(&[
            AsRef::<jit::VarRef>::as_ref(&self.x).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.y).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.z).clone(),
        ])
    }
}
impl<T: jit::AsVarType> Vector4<T> {
    pub fn ravel(&self) -> jit::VarRef {
        jit::vec(&[
            AsRef::<jit::VarRef>::as_ref(&self.x).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.y).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.z).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.w).clone(),
        ])
    }
}

impl<T: jit::AsVarType> Vector2<T>
where
    Vector2<T>: Scatter,
{
    pub fn scatter(&self, dst: impl AsRef<Self>, index: impl AsRef<Var<u32>>) {
        Scatter::scatter(self, dst, index)
    }
    pub fn scatter_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    ) {
        Scatter::scatter_if(self, dst, index, condition)
    }
}

impl<T: jit::AsVarType> Vector3<T>
where
    Vector3<T>: Scatter,
{
    pub fn scatter(&self, dst: impl AsRef<Self>, index: impl AsRef<Var<u32>>) {
        Scatter::scatter(self, dst, index)
    }
    pub fn scatter_if(
        &self,
        dst: impl AsRef<Self>,
        index: impl AsRef<Var<u32>>,
        condition: impl AsRef<Var<bool>>,
    ) {
        Scatter::scatter_if(self, dst, index, condition)
    }
}