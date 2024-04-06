use hephaestus as hep;

use crate::bsdf::BSDF;

#[derive(hep::Traverse)]
pub struct SceneGeometry {
    triangles: hep::Vector3u,
    vertices: hep::Vector3f,
}

#[derive(hep::Traverse)]
pub struct Scene {
    bsdfs: Vec<Box<dyn BSDF>>,
    geometries: Vec<SceneGeometry>,
    instances: hep::Var<jit::Instance>,
}

impl Scene {}
