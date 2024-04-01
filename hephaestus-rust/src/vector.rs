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

// pub type Vector2<T> = Var<mint::Vector2<T>>;
// pub type Vector3<T> = Var<mint::Vector3<T>>;
// pub type Vector4<T> = Var<mint::Vector4<T>>;
//
//
// pub fn vec2<T: jit::AsVarType>(x: impl AsRef<Var<T>>, y: impl AsRef<Var<T>>) -> Vector2<T> {
//     jit::vec(&[x.as_ref().0.clone(), y.as_ref().0.clone()]).into()
// }
// pub fn vec3<T: jit::AsVarType>(
//     x: impl AsRef<Var<T>>,
//     y: impl AsRef<Var<T>>,
//     z: impl AsRef<Var<T>>,
// ) -> Vector3<T> {
//     jit::vec(&[
//         x.as_ref().0.clone(),
//         y.as_ref().0.clone(),
//         z.as_ref().0.clone(),
//     ])
//     .into()
// }
// pub fn vec4<T: jit::AsVarType>(
//     x: impl AsRef<Var<T>>,
//     y: impl AsRef<Var<T>>,
//     z: impl AsRef<Var<T>>,
//     w: impl AsRef<Var<T>>,
// ) -> Vector4<T> {
//     jit::vec(&[
//         x.as_ref().0.clone(),
//         y.as_ref().0.clone(),
//         z.as_ref().0.clone(),
//         w.as_ref().0.clone(),
//     ])
//     .into()
// }
//
// impl<T: jit::AsVarType> Vector2<T> {
//     pub fn dot(&self, other: impl AsRef<Self>) -> Var<T> {
//         self.0.inner(&other.as_ref().0).into()
//     }
// }
// impl<T: jit::AsVarType> Vector3<T> {
//     pub fn dot(&self, other: impl AsRef<Self>) -> Var<T> {
//         self.0.inner(&other.as_ref().0).into()
//     }
// }
// impl<T: jit::AsVarType> Vector4<T> {
//     pub fn dot(&self, other: impl AsRef<Self>) -> Var<T> {
//         self.0.inner(&other.as_ref().0).into()
//     }
// }
//
// impl<T: jit::AsVarType> Vector2<T> {
//     pub fn x(&self) -> Var<T> {
//         self.0.extract(0).into()
//     }
//     pub fn y(&self) -> Var<T> {
//         self.0.extract(1).into()
//     }
//     pub fn xy(&self) -> Vector2<T> {
//         vec2(self.x(), self.y())
//     }
//     pub fn yx(&self) -> Vector2<T> {
//         vec2(self.y(), self.x())
//     }
// }
// impl<T: jit::AsVarType> Vector3<T> {
//     pub fn x(&self) -> Var<T> {
//         self.0.extract(0).into()
//     }
//     pub fn y(&self) -> Var<T> {
//         self.0.extract(1).into()
//     }
//     pub fn z(&self) -> Var<T> {
//         self.0.extract(2).into()
//     }
//
//     pub fn xy(&self) -> Vector2<T> {
//         vec2(self.x(), self.y())
//     }
//     pub fn yx(&self) -> Vector2<T> {
//         vec2(self.y(), self.x())
//     }
//     pub fn xz(&self) -> Vector2<T> {
//         vec2(self.x(), self.z())
//     }
//     pub fn zx(&self) -> Vector2<T> {
//         vec2(self.z(), self.x())
//     }
//     pub fn zy(&self) -> Vector2<T> {
//         vec2(self.z(), self.y())
//     }
//
//     pub fn xyz(&self) -> Vector3<T> {
//         vec3(self.x(), self.y(), self.z())
//     }
//     pub fn xzy(&self) -> Vector3<T> {
//         vec3(self.x(), self.z(), self.y())
//     }
//     pub fn yxz(&self) -> Vector3<T> {
//         vec3(self.y(), self.x(), self.z())
//     }
//     pub fn yzx(&self) -> Vector3<T> {
//         vec3(self.y(), self.z(), self.x())
//     }
//     pub fn zxy(&self) -> Vector3<T> {
//         vec3(self.z(), self.x(), self.y())
//     }
//     pub fn zyx(&self) -> Vector3<T> {
//         vec3(self.z(), self.y(), self.x())
//     }
// }
// impl<T: jit::AsVarType> Vector4<T> {
//     pub fn x(&self) -> Var<T> {
//         self.0.extract(0).into()
//     }
//     pub fn y(&self) -> Var<T> {
//         self.0.extract(1).into()
//     }
//     pub fn z(&self) -> Var<T> {
//         self.0.extract(2).into()
//     }
//     pub fn w(&self) -> Var<T> {
//         self.0.extract(3).into()
//     }
// }
//
// // Implement op traits for vectors
//
// macro_rules! uop {
//     ($op:ident for $($types:ident),*) => {
//         uop!($op -> (Self) for $($types),*);
//     };
//     ($op:ident -> ($ret_type:ty) for $($vectors:ident),*) => {
//         paste::paste!{
//             $(
//                 impl<T: jit::AsVarType> var::[<$op:camel>] for $vectors<T>
//                     where Var<T>: var::[<$op:camel>]
//                 {
//                     type Return = $ret_type;
//                     fn $op(&self) -> Self::Return {
//                         self.0.$op().into()
//                     }
//                 }
//             )*
//         }
//     };
// }
//
// uop!(neg for Vector2, Vector3, Vector4);
// uop!(sqrt for Vector2, Vector3, Vector4);
// uop!(abs for Vector2, Vector3, Vector4);
// uop!(sin for Vector2, Vector3, Vector4);
// uop!(cos for Vector2, Vector3, Vector4);
// uop!(exp2 for Vector2, Vector3, Vector4);
// uop!(log2 for Vector2, Vector3, Vector4);
//
// macro_rules! bop {
//     ($op:ident for $($types:ident),*) => {
//         bop!($op (self, Self) -> (Self) for $($types),*);
//     };
//     ($op:ident -> ($ret_type:ty) for $($types:ident),*) => {
//         bop!($op (self, Self) -> ($ret_type) for $($types),*);
//     };
//     ($op:ident (self, $rhs:ty) -> ($ret_type:ty) for $($vectors:ident),*) => {
//         paste::paste!{
//             $(
//                 impl<T: jit::AsVarType> var::[<$op:camel>] for $vectors<T>
//                     where Var<T>: var::[<$op:camel>]
//                 {
//                     type Return = $ret_type;
//                     type Rhs = $rhs;
//                     fn $op(&self, other: impl AsRef<Self::Rhs>) -> Self::Return {
//                         self.0.$op(&other.as_ref().0).into()
//                     }
//                 }
//             )*
//         }
//     };
// }
//
// // Arithmetic
// bop!(add for Vector2, Vector3, Vector4);
// bop!(sub for Vector2, Vector3, Vector4);
// bop!(mul for Vector2, Vector3, Vector4);
// bop!(div for Vector2, Vector3, Vector4);
// bop!(modulus for Vector2, Vector3, Vector4);
// bop!(min for Vector2, Vector3, Vector4);
// bop!(max for Vector2, Vector3, Vector4);
//
// bop!(and for Vector2, Vector3, Vector4);
// bop!(or for Vector2, Vector3, Vector4);
// bop!(xor for Vector2, Vector3, Vector4);
//
// bop!(eq -> (Var<bool>) for Vector2, Vector3, Vector4);
// bop!(neq -> (Var<bool>) for Vector2, Vector3, Vector4);
// bop!(lt -> (Var<bool>) for Vector2, Vector3, Vector4);
// bop!(le -> (Var<bool>) for Vector2, Vector3, Vector4);
// bop!(gt -> (Var<bool>) for Vector2, Vector3, Vector4);
// bop!(ge -> (Var<bool>) for Vector2, Vector3, Vector4);
//
// macro_rules! top {
//     ($op:ident for $($types:ident),*) => {
//         top!($op (self, Self, Self) -> (Self) for $($types),*);
//     };
//     ($op:ident -> ($ret_type:ty) for $($types:ident),*) => {
//         top!($op (self, Self, Self) -> ($ret_type) for $($types),*);
//     };
//     ($op:ident (self, $b:ty, $c:ty) -> ($ret_type:ty) for $($vectors:ident),*) => {
//         paste::paste!{
//             $(
//                 impl<T: jit::AsVarType> var::[<$op:camel>] for $vectors<T>
//                     where Var<T>: var::[<$op:camel>]
//                 {
//                     type Return = $ret_type;
//                     fn $op(&self, b: impl AsRef<$b>, c: impl AsRef<$c>) -> Self::Return {
//                         self.0.$op(&b.as_ref().0, &c.as_ref().0).into()
//                     }
//                 }
//             )*
//         }
//     };
// }
//
// top!(fma for Vector2, Vector3, Vector4);
