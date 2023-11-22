use super::presenter::{Presenter, PresenterInfo};
use super::*;
use crate::backend;
use crate::*;

#[test]
fn image() {
    let device = backend::Device::vulkan(0).unwrap();

    let tex = device.create_texture([100, 100, 100], 4).unwrap();

    dbg!(&tex);
}

#[test]
fn presenter_accel() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let x = tr::array(&[1f32, 0f32, 1f32], &device);
    let y = tr::array(&[0f32, 1f32, 1f32], &device);
    let z = tr::array(&[0f32, 0f32, 0f32], &device);

    let vertices = tr::vec(&[&x, &y, &z]);
    let triangles = tr::vec(&[
        &tr::sized_literal(0u32, 1),
        &tr::literal(1u32),
        &tr::literal(2u32),
    ]);

    let desc = tr::AccelDesc {
        geometries: vec![tr::GeometryDesc::Triangles {
            triangles,
            vertices,
        }],
        instances: vec![tr::InstanceDesc {
            geometry: 0,
            transform: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
            ],
        }],
    };

    let accel = tr::accel(&desc);

    accel.schedule();

    let x = tr::array(&[0.6f32, 0.3f32], &device);
    let y = tr::array(&[0.6f32, 0.3f32], &device);
    let z = tr::array(&[0.1f32, 0.1f32], &device);

    let o = tr::vec(&[&x, &y, &z]);
    let d = tr::vec(&[&tr::literal(0f32), &tr::literal(0f32), &tr::literal(-1f32)]);
    let tmin = tr::literal(0f32);
    let tmax = tr::literal(10_000f32);

    // d.schedule();
    // o.schedule();

    let intersection_ty = accel.trace_ray(&o, &d, &tmin, &tmax);
    intersection_ty.schedule();

    let graph = tr::compile();

    let info = PresenterInfo {
        width: 800,
        height: 600,
    };
    let presenter = Presenter::create(&device.as_vulkan().unwrap(), &info);

    presenter.render_loop(|| {
        graph.launch(&device);
    });

    dbg!(intersection_ty.to_vec::<i32>());
    // dbg!(o.to_vec::<f32>());
    // dbg!(d.to_vec::<f32>());
}
