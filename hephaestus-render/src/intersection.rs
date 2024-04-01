use hephaestus as hep;

#[derive(hep::Traverse, hep::Construct)]
pub struct PreliminaryIntersection {
    pub bx: hep::Float32,
    pub by: hep::Float32,
    pub instance_id: hep::UInt32,
    pub primitive_id: hep::UInt32,
    pub valid: hep::UInt32,
}

#[derive(hep::Traverse, hep::Construct)]
pub struct SurfaceIntersection {
    preliminary: PreliminaryIntersection,
}
