use hephaestus as hep;

use crate::bsdf::BSDF;
use crate::instance::Instance;
use crate::intersection::PreliminaryIntersection;
use crate::ray::Ray3f;

#[derive(hep::Traverse, Hash)]
pub struct SceneGeometry {
    pub triangles: hep::Vector3u,
    pub vertices: hep::Vector3f,
}

#[derive(hep::Traverse, Hash)]
pub struct Scene {
    bsdfs: Vec<Box<dyn BSDF>>,

    geometries: Vec<SceneGeometry>,
    instances: Instance,

    accel: hep::Accel,
}

impl From<SceneDesc> for Scene {
    fn from(desc: SceneDesc) -> Self {
        let geometries = desc
            .geometries
            .iter()
            .map(|g| hep::GeometryDesc::Triangles {
                triangles: g.triangles.clone(),
                vertices: g.vertices.clone(),
            })
            .collect::<Vec<_>>();

        let accel_desc = hep::AccelDesc {
            geometries: &geometries,
            instances: desc.instances.clone().into(),
        };
        Self {
            bsdfs: desc.bsdfs,
            geometries: desc.geometries,
            instances: desc.instances,
            accel: hep::Accel::create(&accel_desc),
        }
    }
}

#[derive(hep::Traverse, Hash)]
pub struct SceneDesc {
    pub bsdfs: Vec<Box<dyn BSDF>>,

    pub geometries: Vec<SceneGeometry>,
    pub instances: Instance,
}

impl Scene {
    pub fn ray_intersect_preliminary(&self, ray: &Ray3f) -> PreliminaryIntersection {
        let pi = self.accel.trace_ray(ray.clone());
        let pi = PreliminaryIntersection::from(pi);
        pi
    }
}

#[cfg(test)]
mod test {
    use crate::instance::Instance;

    use super::*;
    use hephaestus as hep;
    #[test]
    fn preliminary_intersection() {
        let device = hep::vulkan(0);

        let x = hep::array(&[1f32, 0f32, 1f32], &device);
        let y = hep::array(&[0f32, 1f32, 1f32], &device);
        let z = hep::array(&[0f32, 0f32, 0f32], &device);
        let vertices = hep::vec3(x, y, z);

        let x = hep::sized_literal(0u32, 1);
        let y = hep::literal(1u32);
        let z = hep::literal(2u32);
        let triangles = hep::vec3(x, y, z);

        let transform = hep::Matrix4f {
            x_axis: hep::vec4(1., 0., 0., 0.),
            y_axis: hep::vec4(0., 1., 0., 0.),
            z_axis: hep::vec4(0., 0., 1., 0.),
            w_axis: hep::vec4(0., 0., 0., 1.),
        };

        let instances = Instance {
            transform,
            geometry: hep::sized_literal(0, 1).into(),
        };

        let desc = SceneDesc {
            bsdfs: vec![],
            geometries: vec![SceneGeometry {
                triangles,
                vertices,
            }],
            instances: instances.into(),
        };

        let scene = desc.into();

        let ray = Ray3f {
            o: hep::point3(hep::sized_literal(0.6, 1), 0.6, 0.1),
            d: hep::vec3(0., 0., -1.),
            tmin: (0.).into(),
            tmax: (10_000.).into(),
        };

        #[hep::recorded]
        fn render(scene: &Scene, ray: &Ray3f) -> PreliminaryIntersection {
            scene.ray_intersect_preliminary(ray)
        }

        let result = render(&device, &scene, &ray).unwrap().0;

        assert_eq!(result.valid.to_vec()[0], true);
    }
}
