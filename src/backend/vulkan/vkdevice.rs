use crate::backend::{self, AccelDesc};
use crate::graph::{Pass, PassOp};
use crate::op::DeviceOp;

use super::accel::Accel;
use super::buffer::Buffer;
use super::context::Context;
use super::{accel, VulkanDevice};

impl VulkanDevice {
    pub fn reduce(&self) {
        todo!()
    }
    pub fn build_accel<'a>(
        &'a self,
        ctx: &mut Context,
        accel_desc: &AccelDesc,
        accel: &Accel,
        buffers: impl IntoIterator<Item = &'a Buffer>,
    ) {
        // WARN: This is potentially very unsafe, since we are just randomly
        // accessing the buffers and hoping for them to be index/vertex
        // buffers

        let mut buffers = buffers.into_iter();

        let instances = &buffers.next().unwrap();

        let geometries = accel_desc
            .geometries
            .iter()
            .map(|g| match g {
                backend::GeometryDesc::Triangles { .. } => {
                    accel::AccelGeometryBuildInfo::Triangles {
                        triangles: &buffers.next().unwrap(),
                        vertices: &buffers.next().unwrap(),
                    }
                }
            })
            .collect::<Vec<_>>();

        let desc = accel::AccelBuildInfo {
            geometries: &geometries,
            instances,
        };

        accel.build(ctx, desc);
    }
}
