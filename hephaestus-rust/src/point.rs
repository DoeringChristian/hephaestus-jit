use super::var;
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

// Ops
macro_rules! uop {
    ($op:ident for $($types:ident),*) => {
        uop!($op -> (Self) for $($types),*);
    };
    ($op:ident -> ($ret_type:ty) for $($vectors:ident),*) => {
        paste::paste!{
            $(
                impl<T: jit::AsVarType> var::[<$op:camel>] for $vectors<T>
                    where Var<T>: var::[<$op:camel>]
                {
                    type Return = $ret_type;
                    fn $op(&self) -> Self::Return {
                        self.0.$op().into()
                    }
                }
            )*
        }
    };
}

uop!(neg for Point2, Point3);
uop!(sqrt for Point2, Point3);
uop!(abs for Point2, Point3);
uop!(sin for Point2, Point3);
uop!(cos for Point2, Point3);
uop!(exp2 for Point2, Point3);
uop!(log2 for Point2, Point3);

macro_rules! bop {
    ($op:ident for $($types:ident),*) => {
        bop!($op (self, Self) -> (Self) for $($types),*);
    };
    ($op:ident -> ($ret_type:ty) for $($types:ident),*) => {
        bop!($op (self, Self) -> ($ret_type) for $($types),*);
    };
    ($op:ident (self, $rhs:ty) -> ($ret_type:ty) for $($vectors:ident),*) => {
        paste::paste!{
            $(
                impl<T: jit::AsVarType> var::[<$op:camel>] for $vectors<T>
                    where Var<T>: var::[<$op:camel>]
                {
                    type Return = $ret_type;
                    type Rhs = $rhs;
                    fn $op(&self, other: impl AsRef<Self::Rhs>) -> Self::Return {
                        self.0.$op(&other.as_ref().0).into()
                    }
                }
            )*
        }
    };
}

// Arithmetic
bop!(add for Point2, Point3);
bop!(sub for Point2, Point3);
bop!(mul for Point2, Point3);
bop!(div for Point2, Point3);
bop!(modulus for Point2, Point3);
bop!(min for Point2, Point3);
bop!(max for Point2, Point3);

bop!(and for Point2, Point3);
bop!(or for Point2, Point3);
bop!(xor for Point2, Point3);

bop!(eq -> (Var<bool>) for Point2, Point3);
bop!(neq -> (Var<bool>) for Point2, Point3);
bop!(lt -> (Var<bool>) for Point2, Point3);
bop!(le -> (Var<bool>) for Point2, Point3);
bop!(gt -> (Var<bool>) for Point2, Point3);
bop!(ge -> (Var<bool>) for Point2, Point3);

macro_rules! top {
    ($op:ident for $($types:ident),*) => {
        top!($op (self, Self, Self) -> (Self) for $($types),*);
    };
    ($op:ident -> ($ret_type:ty) for $($types:ident),*) => {
        top!($op (self, Self, Self) -> ($ret_type) for $($types),*);
    };
    ($op:ident (self, $b:ty, $c:ty) -> ($ret_type:ty) for $($vectors:ident),*) => {
        paste::paste!{
            $(
                impl<T: jit::AsVarType> var::[<$op:camel>] for $vectors<T>
                    where Var<T>: var::[<$op:camel>]
                {
                    type Return = $ret_type;
                    fn $op(&self, b: impl AsRef<$b>, c: impl AsRef<$c>) -> Self::Return {
                        self.0.$op(&b.as_ref().0, &c.as_ref().0).into()
                    }
                }
            )*
        }
    };
}

top!(fma for Point2, Point3);
