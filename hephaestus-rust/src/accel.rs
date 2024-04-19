use std::marker::PhantomData;

use jit;

use crate::var::*;
use crate::vector::*;

#[derive(Debug, Clone)]
pub enum GeometryDesc {
    Triangles {
        triangles: Vector3u,
        vertices: Vector3f,
    },
}

#[derive(Debug, Clone)]
pub struct AccelDesc<'a> {
    pub geometries: &'a [GeometryDesc],
    pub instances: Var<jit::Instance>,
}

#[derive(jit::Traverse, jit::Construct, Hash)]
pub struct Accel {
    accel: jit::VarRef,
}

impl Accel {
    pub fn create(desc: &AccelDesc) -> Self {
        let geometries = desc
            .geometries
            .iter()
            .map(|g| match g {
                GeometryDesc::Triangles {
                    triangles,
                    vertices,
                } => {
                    let triangles = triangles.ravel();
                    let vertices = vertices.ravel();
                    dbg!(triangles.extent());
                    dbg!(vertices.extent());
                    jit::GeometryDesc::Triangles {
                        triangles,
                        vertices,
                    }
                }
            })
            .collect::<Vec<_>>();

        let accel = jit::accel(&jit::AccelDesc {
            geometries: &geometries,
            instances: desc.instances.0.clone(),
        });
        Self { accel }
    }
    pub fn trace_ray(&self, ray: impl Into<Var<jit::Ray3f>>) -> Var<jit::Intersection> {
        Var(self.accel.trace_ray(&ray.into()), PhantomData)
    }
}

#[cfg(test)]
mod test {
    use crate as hep;
    #[test]
    fn trace_ray() {
        #[hep::recorded]
        fn render(
            scene: &TriangleScene,
            ray: &hep::Var<jit::Ray3f>,
        ) -> hep::Var<jit::Intersection> {
            let geometries = scene
                .triangles
                .iter()
                .zip(scene.vertices.iter())
                .map(|(triangles, vertices)| hep::GeometryDesc::Triangles {
                    triangles: triangles.clone(),
                    vertices: vertices.clone(),
                })
                .collect::<Vec<_>>();

            let desc = hep::AccelDesc {
                geometries: &geometries,
                instances: scene.instances.clone(),
            };

            let accel = hep::Accel::create(&desc);

            accel.trace_ray(ray)
        }

        let device = hep::vulkan(0);

        let x = hep::array(&[1f32, 0f32, 1f32], &device);
        let y = hep::array(&[0f32, 1f32, 1f32], &device);
        let z = hep::array(&[0f32, 0f32, 0f32], &device);
        let vertices = hep::vec3(x, y, z);

        let x = hep::sized_literal(0u32, 1);
        let y = hep::literal(1u32);
        let z = hep::literal(2u32);
        let triangles = hep::vec3(x, y, z);

        let instances = hep::array(
            &[jit::Instance {
                transform: [
                    1f32, 0f32, 0f32, 0f32, //
                    0f32, 1f32, 0f32, 0f32, //
                    0f32, 0f32, 1f32, 0f32, //
                ],
                geometry: 0,
            }],
            &device,
        );

        #[derive(hep::Traverse, hep::Construct, Hash)]
        pub struct TriangleScene {
            triangles: Vec<hep::Vector3u>,
            vertices: Vec<hep::Vector3f>,
            instances: hep::Var<jit::Instance>,
        }

        let scene = TriangleScene {
            triangles: vec![triangles],
            vertices: vec![vertices],
            instances,
        };

        let ray = hep::array(
            &[
                jit::Ray3f {
                    o: mint::Point3 {
                        x: 0.6,
                        y: 0.6,
                        z: 0.1,
                    },
                    d: mint::Vector3 {
                        x: 0.,
                        y: 0.,
                        z: -1.,
                    },
                    tmin: 0.,
                    tmax: 10_000.,
                },
                jit::Ray3f {
                    o: mint::Point3 {
                        x: 0.3,
                        y: 0.3,
                        z: 0.1,
                    },
                    d: mint::Vector3 {
                        x: 0.,
                        y: 0.,
                        z: -1.,
                    },
                    tmin: 0.,
                    tmax: 10_000.,
                },
            ],
            &device,
        );

        let intersection = render(&device, &scene, &ray).unwrap().0;

        insta::assert_debug_snapshot!(intersection.to_vec());

        dbg!(intersection.to_vec());
    }
}
