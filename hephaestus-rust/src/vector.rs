use super::var::Var;

pub fn vec2<T: jit::AsVarType>(x: impl AsRef<Var<T>>, y: impl AsRef<Var<T>>) -> Vector2<T> {
    jit::vec(&[x.as_ref().0.clone(), y.as_ref().0.clone()]).into()
}
pub fn vec3<T: jit::AsVarType>(
    x: impl AsRef<Var<T>>,
    y: impl AsRef<Var<T>>,
    z: impl AsRef<Var<T>>,
) -> Vector3<T> {
    jit::vec(&[
        x.as_ref().0.clone(),
        y.as_ref().0.clone(),
        z.as_ref().0.clone(),
    ])
    .into()
}
pub fn vec4<T: jit::AsVarType>(
    x: impl AsRef<Var<T>>,
    y: impl AsRef<Var<T>>,
    z: impl AsRef<Var<T>>,
    w: impl AsRef<Var<T>>,
) -> Vector4<T> {
    jit::vec(&[
        x.as_ref().0.clone(),
        y.as_ref().0.clone(),
        z.as_ref().0.clone(),
        w.as_ref().0.clone(),
    ])
    .into()
}

impl<T: jit::AsVarType> Vector2<T> {
    pub fn dot(&self, other: impl AsRef<Self>) -> Var<T> {
        self.0.dot(&other.as_ref().0).into()
    }
}
impl<T: jit::AsVarType> Vector3<T> {
    pub fn dot(&self, other: impl AsRef<Self>) -> Var<T> {
        self.0.dot(&other.as_ref().0).into()
    }
}
impl<T: jit::AsVarType> Vector4<T> {
    pub fn dot(&self, other: impl AsRef<Self>) -> Var<T> {
        self.0.dot(&other.as_ref().0).into()
    }
}

impl<T: jit::AsVarType> Vector2<T> {
    pub fn x(&self) -> Var<T> {
        self.0.extract(0).into()
    }
    pub fn y(&self) -> Var<T> {
        self.0.extract(1).into()
    }
    pub fn xy(&self) -> Vector2<T> {
        vec2(self.x(), self.y())
    }
    pub fn yx(&self) -> Vector2<T> {
        vec2(self.y(), self.x())
    }
}
impl<T: jit::AsVarType> Vector3<T> {
    pub fn x(&self) -> Var<T> {
        self.0.extract(0).into()
    }
    pub fn y(&self) -> Var<T> {
        self.0.extract(1).into()
    }
    pub fn z(&self) -> Var<T> {
        self.0.extract(2).into()
    }

    pub fn xy(&self) -> Vector2<T> {
        vec2(self.x(), self.y())
    }
    pub fn yx(&self) -> Vector2<T> {
        vec2(self.y(), self.x())
    }
    pub fn xz(&self) -> Vector2<T> {
        vec2(self.x(), self.z())
    }
    pub fn zx(&self) -> Vector2<T> {
        vec2(self.z(), self.x())
    }
    pub fn zy(&self) -> Vector2<T> {
        vec2(self.z(), self.y())
    }

    pub fn xyz(&self) -> Vector3<T> {
        vec3(self.x(), self.y(), self.z())
    }
    pub fn xzy(&self) -> Vector3<T> {
        vec3(self.x(), self.z(), self.y())
    }
    pub fn yxz(&self) -> Vector3<T> {
        vec3(self.y(), self.x(), self.z())
    }
    pub fn yzx(&self) -> Vector3<T> {
        vec3(self.y(), self.z(), self.x())
    }
    pub fn zxy(&self) -> Vector3<T> {
        vec3(self.z(), self.x(), self.y())
    }
    pub fn zyx(&self) -> Vector3<T> {
        vec3(self.z(), self.y(), self.x())
    }
}
impl<T: jit::AsVarType> Vector4<T> {
    pub fn x(&self) -> Var<T> {
        self.0.extract(0).into()
    }
    pub fn y(&self) -> Var<T> {
        self.0.extract(1).into()
    }
    pub fn z(&self) -> Var<T> {
        self.0.extract(2).into()
    }
    pub fn w(&self) -> Var<T> {
        self.0.extract(3).into()
    }
}

pub type Vector2<T> = Var<mint::Vector2<T>>;
pub type Vector3<T> = Var<mint::Vector3<T>>;
pub type Vector4<T> = Var<mint::Vector4<T>>;

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
