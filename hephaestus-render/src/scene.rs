use hep::traits::Select;
use hep::Gather;
use hephaestus as hep;

use crate::bsdf::BSDF;
use crate::instance::Instance;
use crate::intersection::{PreliminaryIntersection, SurfaceIntersection};
use crate::ray::Ray3f;

#[derive(hep::Traverse, Hash)]
pub struct SceneGeometry {
    pub triangles: hep::Vector3u,
    pub vertices: hep::Vector3f,
}

#[derive(hep::Traverse, Hash)]
pub struct Scene {
    pub bsdfs: Vec<Box<dyn BSDF>>,

    pub geometries: Vec<SceneGeometry>,
    pub instances: Instance,

    pub accel: hep::Accel,
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

        let intersection_type: hep::UInt32 = pi.extract(4).into();
        let valid = hep::literal(false).select(intersection_type.eq(0u32), true);
        PreliminaryIntersection {
            bx: pi.extract(0).into(),
            by: pi.extract(1).into(),
            wi: hep::vec3(ray.d.x.neg(), ray.d.y.neg(), ray.d.z.neg()),
            instance_id: pi.extract(2).into(),
            primitive_id: pi.extract(3).into(),
            valid,
        }
    }
    pub fn ray_intersect(&self, ray: &Ray3f) -> SurfaceIntersection {
        self.ray_intersect_preliminary(ray).finalize(self)
    }
    pub fn ray_test(&self, ray: &Ray3f) -> hep::Mask {
        self.ray_intersect_preliminary(ray).valid
    }
    pub fn eval_bsdf(&self, si: &SurfaceIntersection, wo: hep::Vector3f) -> hep::Vector3f {
        let mut result = hep::vec3(0., 0., 0.);
        for (i, bsdf) in self.bsdfs.iter().enumerate() {
            let mask = si.bsdf.eq(i as u32);
            let bsdf_res = bsdf.eval(si.wi.clone(), wo.clone(), mask.clone());
            result = bsdf_res.select(mask.clone(), result.clone());
        }
        return result;
    }
}

#[cfg(test)]
mod test {
    use crate::bsdf::DiffuseBSDF;
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

        let one = hep::sized_literal(1.0, 1);
        let zero = hep::sized_literal(0.0, 1);

        let transform = hep::Matrix4f {
            x_axis: hep::vec4(&one, &zero, &zero, &zero),
            y_axis: hep::vec4(&zero, &one, &zero, &zero),
            z_axis: hep::vec4(&zero, &zero, &one, &zero),
            w_axis: hep::vec4(&zero, &zero, &zero, &one),
        };

        let instances = Instance {
            transform,
            geometry: hep::sized_literal(0, 1).into(),
            bsdf: hep::sized_literal(0, 1),
        };

        let desc = SceneDesc {
            bsdfs: vec![Box::new(DiffuseBSDF {
                albedo: hep::vec3(hep::sized_literal(1., 1), 0., 0.),
            })],
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
        fn render(scene: &Scene, ray: &Ray3f) -> hep::Vector3f {
            let si = scene.ray_intersect(ray);
            scene.eval_bsdf(&si, hep::vec3(1., 0., 0.))
        }

        let result = render(&device, &scene, &ray).unwrap().0;

        dbg!(result.x.to_vec());
        dbg!(result.y.to_vec());
        dbg!(result.z.to_vec());
    }
}
