use hephaestus as hep;
use jit::Traverse;

#[derive(Clone, hep::Traverse, hep::Construct, Hash)]
pub struct Ray3f {
    pub o: hep::Point3f,
    pub d: hep::Vector3f,
    pub tmin: hep::Float32,
    pub tmax: hep::Float32,
}

impl From<&Ray3f> for Ray3f {
    fn from(value: &Ray3f) -> Self {
        value.clone()
    }
}

impl From<Ray3f> for hep::Var<jit::Ray3f> {
    fn from(value: Ray3f) -> Self {
        value.ravel().into()
    }
}
