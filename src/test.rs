use crate::vartype::Instance;
use crate::{backend, tr, vulkan};

#[test]
fn simple1() {
    pretty_env_logger::try_init().ok();

    let device = backend::Device::vulkan(0).unwrap();

    let i = tr::index(10);
    let j = tr::index(5);

    j.add(&tr::literal(1u32)).scatter(&i, &j);

    j.schedule();

    tr::with_trace(|trace| {
        dbg!(&trace);
    });
    tr::SCHEDULE.with(|s| {
        dbg!(&s);
    });

    let graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    dbg!(graph.n_passes());

    dbg!(i.to_vec::<u32>());
    dbg!(j.to_vec::<u32>());
    assert_eq!(i.to_vec::<u32>(), vec![1, 2, 3, 4, 5, 5, 6, 7, 8, 9]);
    assert_eq!(j.to_vec::<u32>(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn simple_u16() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let c = tr::sized_literal(1u16, 10);
    c.schedule();

    tr::with_trace(|trace| {
        dbg!(&trace);
    });
    tr::SCHEDULE.with(|s| {
        dbg!(&s);
    });
    let graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    dbg!(c.to_vec::<u16>());
    assert_eq!(c.to_vec::<u16>(), vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
}

#[test]
fn scatter_chain1() {
    let device = backend::Device::vulkan(0).unwrap();

    let b0 = tr::sized_literal(0, 5);

    tr::literal(1).scatter(&b0, &tr::index(10));

    let b1 = b0.add(&tr::literal(1));
    b1.schedule();

    tr::with_trace(|trace| {
        dbg!(&trace);
    });
    tr::SCHEDULE.with(|s| {
        dbg!(&s);
    });

    let graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    // dbg!(&b1.data().buffer().unwrap().to_host::<i32>().unwrap());
    assert_eq!(b1.to_vec::<i32>(), vec![2, 2, 2, 2, 2]);
}
#[test]
fn scatter_chain2() {
    let device = backend::Device::vulkan(0).unwrap();

    let a = tr::sized_literal(0, 5);
    let b = a.add(&tr::literal(1));
    tr::literal(1).scatter(&a, &tr::index(5));

    b.schedule();

    let graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    dbg!(b.to_vec::<i32>());
    dbg!(a.to_vec::<i32>());
    assert_eq!(b.to_vec::<i32>(), vec![2, 2, 2, 2, 2]);
    assert_eq!(a.to_vec::<i32>(), vec![1, 1, 1, 1, 1]);
}
#[test]
fn extract() {
    let device = backend::Device::vulkan(0).unwrap();

    let a = tr::sized_literal(1f32, 10);
    let b = tr::sized_literal(2f32, 10);

    let v = tr::vec(&[&a, &b]);

    let a = v.extract(0);
    let b = v.extract(1);

    v.schedule();
    a.schedule();
    b.schedule();

    tr::compile().launch(&device);

    dbg!(v.to_vec::<f32>());
    dbg!(a.to_vec::<f32>());
    dbg!(b.to_vec::<f32>());
}
#[test]
fn extract2() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let b = tr::sized_literal(0xffu8, 2);
    let a = tr::sized_literal(2u32, 2);

    let s = tr::composite(&[&a, &b]);

    s.schedule();

    tr::compile().launch(&device);

    dbg!(&s.to_vec::<u8>());
}
#[test]
fn test_struct() {
    let device = backend::Device::vulkan(0).unwrap();

    let a = tr::sized_literal(1u8, 10);
    let b = tr::sized_literal(2u32, 10);

    let s = tr::composite(&[&a, &b]);

    let a = s.extract(0);
    let b = s.extract(1);

    s.schedule();
    a.schedule();
    b.schedule();

    tr::compile().launch(&device);

    dbg!(s.to_vec::<u8>());
    dbg!(a.to_vec::<u8>());
    dbg!(b.to_vec::<u32>());
}

#[test]
fn texture() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let b = tr::sized_literal(1f32, 10 * 10 * 4);

    let tex = b.texture(&[10, 10], 4);

    let x = tr::sized_literal(0.5f32, 2);
    let y = tr::sized_literal(0.5f32, 2);
    // let z = tr::sized_literal(0.5f32, 2);

    let v = tex.tex_lookup(&[&x, &y]);

    v.schedule();

    tr::compile().launch(&device);

    tr::with_trace(|trace| {
        dbg!(&trace);
    });
    dbg!(v.to_vec::<f32>());
    assert_eq!(v.to_vec::<f32>(), vec![1.0; 8]);
}
#[test]
fn conditionals() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let dst = tr::sized_literal(true, 100);

    dst.schedule();

    let graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    dbg!(&dst.to_vec::<u8>());
}
#[test]
fn conditional_scatter() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let dst = tr::sized_literal(0, 10);
    let active = tr::array(
        &[
            true, true, false, false, true, false, true, false, true, false,
        ],
        &device,
    );
    dbg!(&active.to_vec::<u8>());

    tr::literal(1).scatter_if(&dst, &tr::index(10), &active);

    dst.schedule();

    let graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    dbg!(&dst.to_vec::<i32>());
}
#[test]
fn select() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let cond = tr::array(&[true, false], &device);

    let true_val = tr::literal(10);
    let false_val = tr::literal(5);

    let res = cond.select(&true_val, &false_val);
    res.schedule();

    let graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    assert_eq!(res.to_vec::<i32>(), vec![10, 5]);
}
#[test]
fn accel() {
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

    let instances = tr::array(
        &[
            1f32, 0f32, 0f32, 0f32, //
            0f32, 1f32, 0f32, 0f32, //
            0f32, 0f32, 1f32, 0f32, //
            0f32, // WARN: This is 0x00000000 in bytes
        ],
        &device,
    );

    let instances = tr::array(
        &[Instance {
            transform: [
                1f32, 0f32, 0f32, 0f32, //
                0f32, 1f32, 0f32, 0f32, //
                0f32, 0f32, 1f32, 0f32, //
            ],
            geometry: 0,
        }],
        &device,
    );

    dbg!(instances.to_vec::<u32>());

    let desc = tr::AccelDesc {
        geometries: vec![tr::GeometryDesc::Triangles {
            triangles,
            vertices,
        }],
        instances,
    };

    let accel = tr::accel(&desc);

    // accel.schedule();

    let x = tr::array(&[0.6f32, 0.3f32], &device);
    let y = tr::array(&[0.6f32, 0.3f32], &device);
    let z = tr::array(&[0.1f32, 0.1f32], &device);

    let o = tr::vec(&[&x, &y, &z]);
    let d = tr::vec(&[
        &tr::sized_literal(0f32, 2),
        &tr::literal(0f32),
        &tr::literal(-1f32),
    ]);
    let tmin = tr::literal(0f32);
    let tmax = tr::literal(10_000f32);

    // d.schedule();
    // o.schedule();

    let intersection_ty = accel.trace_ray(&o, &d, &tmin, &tmax);
    intersection_ty.schedule();

    tr::with_trace(|trace| {
        dbg!(&trace);
    });
    tr::SCHEDULE.with(|s| {
        let s = s.borrow();
        dbg!(&s);
        for v in s.vars.iter() {
            tr::with_trace(|trace| {
                dbg!(trace.var(v.id()));
            })
        }
    });

    let graph = tr::compile();
    graph.launch(&device);

    dbg!(intersection_ty.to_vec::<i32>());
    assert_eq!(intersection_ty.to_vec::<i32>(), vec![1, 0]);
    // dbg!(o.to_vec::<f32>());
    // dbg!(d.to_vec::<f32>());
}
#[test]
fn max() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let x = (100..200).rev().map(|i| i as f32).collect::<Vec<_>>();
    let x = tr::array(&x, &device);

    let max = x.max();

    max.schedule();

    let graph = tr::compile();
    // insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    dbg!(&max.to_vec::<f32>());
}
