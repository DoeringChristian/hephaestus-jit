use hephaestus as hep;

#[derive(hep::Traverse)]
pub struct SceneGeometry {
    triangles: hep::Vector3u,
    vertices: hep::Vector3f,
}

#[derive(hep::Traverse)]
pub struct Scene {
    geometries: Vec<SceneGeometry>,
    instances: hep::Instance,
}

impl Scene {}
