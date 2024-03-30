use jit;

use crate::var::*;
use crate::{Instance, Intersection};

#[derive(Debug, Clone)]
pub enum GeometryDesc {
    Triangles {
        triangles: Vector3u,
        vertices: Vector3f,
    },
}

pub struct AccelDesc<'a> {
    pub geometries: &'a [GeometryDesc],
    pub instances: Var<Instance>,
}

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
                } => jit::GeometryDesc::Triangles {
                    triangles: triangles.0.clone(),
                    vertices: vertices.0.clone(),
                },
            })
            .collect::<Vec<_>>();

        let accel = jit::accel(&jit::AccelDesc {
            geometries: &geometries,
            instances: desc.instances.0.clone(),
        });
        Self { accel }
    }
    // pub fn trace_ray(&self, ray:)
}
