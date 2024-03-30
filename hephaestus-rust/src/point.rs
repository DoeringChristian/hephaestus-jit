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
) -> Point2<T> {
    jit::vec(&[
        x.as_ref().0.clone(),
        y.as_ref().0.clone(),
        z.as_ref().0.clone(),
    ])
    .into()
}
