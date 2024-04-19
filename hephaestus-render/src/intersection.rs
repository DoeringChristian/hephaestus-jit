use hep::*;
use hephaestus as hep;
use hephaestus::traits::*;

use crate::scene::Scene;
use crate::spectrum::Spectrum;

#[derive(Clone, hep::Traverse, hep::Construct)]
pub struct PreliminaryIntersection {
    pub bx: Float32,
    pub by: Float32,
    pub wi: Vector3f,
    pub instance_id: UInt32,
    pub primitive_id: UInt32,
    pub valid: Mask,
}

impl PreliminaryIntersection {
    pub fn finalize(self, scene: &Scene) -> SurfaceIntersection {
        let instance = scene.instances.gather_if(&self.instance_id, &self.valid);
        SurfaceIntersection {
            bx: self.bx,
            by: self.by,
            wi: self.wi,
            instance_id: self.instance_id,
            primitive_id: self.primitive_id,
            valid: self.valid,
            bsdf: instance.bsdf.clone(),
        }
    }
}

#[derive(Clone, Hash, Traverse, Construct)]
pub struct SurfaceIntersection {
    pub bx: Float32,
    pub by: Float32,
    pub wi: Vector3f,
    pub instance_id: UInt32,
    pub primitive_id: UInt32,
    pub valid: Mask,
    pub bsdf: UInt32,
}
