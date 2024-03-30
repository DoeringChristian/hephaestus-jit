use super::Var;

pub type Point2<T> = Var<mint::Point2<T>>;
pub type Point3<T> = Var<mint::Point3<T>>;

pub type Point2f = Point2<f32>;
pub type Point3f = Point3<f32>;

pub type Point2d = Point2<f64>;
pub type Point3d = Point3<f64>;

pub type Point2i = Point2<i32>;
pub type Point3i = Point3<i32>;

pub type Point2u = Point2<u32>;
pub type Point3u = Point3<u32>;

pub fn point2<T: jit::AsVarType>(x: impl AsRef<Var<T>>, y: impl AsRef<Var<T>>) -> Point2<T> {
    jit::vec(&[x.as_ref().0.clone(), y.as_ref().0.clone()]).into()
}
pub fn point3<T: jit::AsVarType>(
    x: impl AsRef<Var<T>>,
    y: impl AsRef<Var<T>>,
    z: impl AsRef<Var<T>>,
) -> Point3<T> {
    jit::vec(&[
        x.as_ref().0.clone(),
        y.as_ref().0.clone(),
        z.as_ref().0.clone(),
    ])
    .into()
}

impl<T: jit::AsVarType> Point2<T> {
    pub fn x(&self) -> Var<T> {
        self.0.extract(0).into()
    }
    pub fn y(&self) -> Var<T> {
        self.0.extract(1).into()
    }
    pub fn xy(&self) -> Point2<T> {
        point2(self.x(), self.y())
    }
    pub fn yx(&self) -> Point2<T> {
        point2(self.y(), self.x())
    }
}
impl<T: jit::AsVarType> Point3<T> {
    pub fn x(&self) -> Var<T> {
        self.0.extract(0).into()
    }
    pub fn y(&self) -> Var<T> {
        self.0.extract(1).into()
    }
    pub fn z(&self) -> Var<T> {
        self.0.extract(2).into()
    }

    pub fn xy(&self) -> Point2<T> {
        point2(self.x(), self.y())
    }
    pub fn yx(&self) -> Point2<T> {
        point2(self.y(), self.x())
    }
    pub fn xz(&self) -> Point2<T> {
        point2(self.x(), self.z())
    }
    pub fn zx(&self) -> Point2<T> {
        point2(self.z(), self.x())
    }
    pub fn zy(&self) -> Point2<T> {
        point2(self.z(), self.y())
    }

    pub fn xyz(&self) -> Point3<T> {
        point3(self.x(), self.y(), self.z())
    }
    pub fn xzy(&self) -> Point3<T> {
        point3(self.x(), self.z(), self.y())
    }
    pub fn yxz(&self) -> Point3<T> {
        point3(self.y(), self.x(), self.z())
    }
    pub fn yzx(&self) -> Point3<T> {
        point3(self.y(), self.z(), self.x())
    }
    pub fn zxy(&self) -> Point3<T> {
        point3(self.z(), self.x(), self.y())
    }
    pub fn zyx(&self) -> Point3<T> {
        point3(self.z(), self.y(), self.x())
    }
}
