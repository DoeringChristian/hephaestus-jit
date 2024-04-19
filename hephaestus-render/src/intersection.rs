use hephaestus as hep;
use hephaestus::traits::*;

#[derive(Clone, hep::Traverse, hep::Construct)]
pub struct PreliminaryIntersection {
    pub bx: hep::Float32,
    pub by: hep::Float32,
    pub instance_id: hep::UInt32,
    pub primitive_id: hep::UInt32,
    pub valid: hep::Mask,
}

impl From<hep::Var<jit::Intersection>> for PreliminaryIntersection {
    fn from(value: hep::Var<jit::Intersection>) -> Self {
        let intersection_type: hep::UInt32 = value.extract(4).into();
        let valid = hep::literal(false).select(intersection_type.eq(0u32), true);

        Self {
            bx: value.extract(0).into(),
            by: value.extract(1).into(),
            instance_id: value.extract(2).into(),
            primitive_id: value.extract(3).into(),
            valid,
        }
    }
}

#[derive(Clone, hep::Traverse, hep::Construct)]
pub struct SurfaceIntersection {
    preliminary: PreliminaryIntersection,
}
