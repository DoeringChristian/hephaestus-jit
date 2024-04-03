use super::Var;

#[derive(Clone, Debug)]
pub struct Point2<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
}

impl<T: jit::AsVarType> jit::Traverse for Point2<T> {
    fn traverse(&self, vars: &mut Vec<jit::VarRef>) -> &'static jit::Layout {
        let layouts = [self.x.traverse(vars), self.y.traverse(vars)];
        jit::Layout::tuple(&layouts)
    }
    fn ravel(&self) -> jit::VarRef {
        jit::vec(&[self.x.0.clone(), self.y.0.clone()])
    }
}

impl<T: jit::AsVarType> jit::Construct for Point2<T> {
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
    fn unravel(var: impl AsRef<jit::VarRef>) -> Self {
        let var = var.as_ref();
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
pub struct Point3<T: jit::AsVarType> {
    x: Var<T>,
    y: Var<T>,
    z: Var<T>,
}

impl<T: jit::AsVarType> jit::Traverse for Point3<T> {
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

impl<T: jit::AsVarType> jit::Construct for Point3<T> {
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
    fn unravel(var: impl AsRef<jit::VarRef>) -> Self {
        let var = var.as_ref();
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

pub type Point2f = Point2<f32>;
pub type Point3f = Point3<f32>;

pub type Point2d = Point2<f64>;
pub type Point3d = Point3<f64>;

pub type Point2i = Point2<i32>;
pub type Point3i = Point3<i32>;

pub type Point2u = Point2<u32>;
pub type Point3u = Point3<u32>;

pub fn point2<T: jit::AsVarType>(
    x: impl AsRef<jit::VarRef>,
    y: impl AsRef<jit::VarRef>,
) -> Point2<T> {
    Point2::<T> {
        x: x.as_ref().into(),
        y: y.as_ref().into(),
    }
}
pub fn point3<T: jit::AsVarType>(
    x: impl AsRef<jit::VarRef>,
    y: impl AsRef<jit::VarRef>,
    z: impl AsRef<jit::VarRef>,
) -> Point3<T> {
    Point3::<T> {
        x: x.as_ref().into(),
        y: y.as_ref().into(),
        z: z.as_ref().into(),
    }
}

impl<T: jit::AsVarType> AsRef<Point2<T>> for Point2<T> {
    fn as_ref(&self) -> &Point2<T> {
        self
    }
}
impl<T: jit::AsVarType> AsRef<Point3<T>> for Point3<T> {
    fn as_ref(&self) -> &Point3<T> {
        self
    }
}

impl<T: jit::AsVarType> From<jit::VarRef> for Point2<T> {
    fn from(value: jit::VarRef) -> Self {
        Self {
            x: value.extract(0).into(),
            y: value.extract(1).into(),
        }
    }
}
impl<T: jit::AsVarType> From<jit::VarRef> for Point3<T> {
    fn from(value: jit::VarRef) -> Self {
        Self {
            x: value.extract(0).into(),
            y: value.extract(1).into(),
            z: value.extract(2).into(),
        }
    }
}

impl<T: jit::AsVarType> Point2<T> {
    pub fn ravel(&self) -> jit::VarRef {
        jit::vec(&[
            AsRef::<jit::VarRef>::as_ref(&self.x).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.y).clone(),
        ])
    }
}
impl<T: jit::AsVarType> Point3<T> {
    pub fn ravel(&self) -> jit::VarRef {
        jit::vec(&[
            AsRef::<jit::VarRef>::as_ref(&self.x).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.y).clone(),
            AsRef::<jit::VarRef>::as_ref(&self.z).clone(),
        ])
    }
}
