use approx::assert_abs_diff_eq;

use crate::vartype::{Instance, Intersection};
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
fn texture2d() {
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
fn texture3d() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let b = tr::sized_literal(1f32, 10 * 10 * 10 * 4);

    let tex = b.texture(&[10, 10, 10], 4);

    let x = tr::sized_literal(0.5f32, 2);
    let y = tr::sized_literal(0.3f32, 2);
    let z = tr::sized_literal(0.6f32, 2);

    let v = tex.tex_lookup(&[&x, &y, &z]);

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

    let intersection = accel.trace_ray(&o, &d, &tmin, &tmax);
    intersection.schedule();

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

    dbg!(intersection.to_vec::<i32>());
    let intersections = intersection.to_vec::<Intersection>();

    assert_abs_diff_eq!(intersections[0].barycentrics[0], 0.4, epsilon = 0.001);
    assert_abs_diff_eq!(intersections[0].barycentrics[1], 0.2, epsilon = 0.001);
    assert!(intersections[0].valid > 0);
    assert_eq!(intersections[0].instance_id, 0);
    assert_eq!(intersections[0].primitive_idx, 0);

    assert_eq!(intersections[1].valid, 0);
}
#[test]
fn reduce_max() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    macro_rules! max_test {
        ($ty:ident, $iter:expr) => {
            let x = ($iter).map(|i| i as $ty).collect::<Vec<_>>();
            let reduced = x.to_vec().into_iter().reduce(|a, b| a.max(b)).unwrap();

            // Launch Kernels:
            let x = tr::array(&x, &device);
            let max = x.reduce_max();
            max.schedule();
            let graph = tr::compile();
            graph.launch(&device);

            assert_eq!(max.to_vec::<$ty>()[0], reduced)
        };
    }

    max_test!(u8, 0..0xff);
    max_test!(i8, -128..127);
    max_test!(f32, 0..100);
}
#[test]
fn reduce_min() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    macro_rules! min_test {
        ($ty:ident, $iter:expr) => {
            let x = ($iter).map(|i| i as $ty).collect::<Vec<_>>();
            let reduced = x.to_vec().into_iter().reduce(|a, b| a.min(b)).unwrap();

            // Launch Kernels:
            let x = tr::array(&x, &device);
            let min = x.reduce_min();
            min.schedule();
            let graph = tr::compile();
            graph.launch(&device);

            assert_eq!(min.to_vec::<$ty>()[0], reduced)
        };
    }

    min_test!(u8, 0..0xff);
    min_test!(i8, -128..127);
    min_test!(i64, -128..127);
    min_test!(u64, 0..0xffff);
    min_test!(f32, 0..100);
}

#[test]
fn reduce_sum() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();
    use rand::Rng;

    macro_rules! rng_test {
        ($ty:ident, $range:expr, $num:expr, $red:expr, $jit_red:ident) => {
            let mut rng = rand::thread_rng();

            let x = (0..$num)
                .map(|_| rng.gen_range($range))
                .map(|i| i as $ty)
                .collect::<Vec<_>>();
            let reduced = x.to_vec().into_iter().reduce(|a, b| $red(a, b)).unwrap();

            // Launch Kernels:
            let x = tr::array(&x, &device);
            let sum = x.$jit_red();
            sum.schedule();
            let graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>()[0], reduced)
        };
    }

    rng_test!(u8, 0..0xff, 1_000, u8::wrapping_add, reduce_sum);
    rng_test!(i8, -128..127, 1_000, i8::wrapping_add, reduce_sum);
    rng_test!(i64, -128..127, 1_000, i64::wrapping_add, reduce_sum);
    rng_test!(u64, 0..0xff, 1_000, u64::wrapping_add, reduce_sum);
    rng_test!(f32, 0..100, 1_000, std::ops::Add::add, reduce_sum);
}

#[test]
fn reduce_prod() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();
    use rand::Rng;

    macro_rules! rng_test {
        ($ty:ident, $range:expr, $num:expr, $red:expr, $jit_red:ident) => {
            let mut rng = rand::thread_rng();

            let x = (0..$num)
                .map(|_| rng.gen_range($range))
                .map(|i| i as $ty)
                .collect::<Vec<_>>();
            let reduced = x.to_vec().into_iter().reduce(|a, b| $red(a, b)).unwrap();

            // Launch Kernels:
            let x = tr::array(&x, &device);
            let sum = x.$jit_red();
            sum.schedule();
            let graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>()[0], reduced)
        };
    }
    macro_rules! rng_test_float {
        ($ty:ident, $range:expr, $num:expr, $map:expr , $red:expr, $jit_red:ident) => {
            let mut rng = rand::thread_rng();

            let x = (0..$num)
                .map(|_| rng.gen_range($range))
                .map(|x| x as $ty)
                .map(|x| $map(x))
                .collect::<Vec<_>>();
            let reduced = x.to_vec().into_iter().reduce(|a, b| $red(a, b)).unwrap();

            // Launch Kernels:
            let x = tr::array(&x, &device);
            let sum = x.$jit_red();
            sum.schedule();
            let graph = tr::compile();
            graph.launch(&device);

            let res = sum.to_vec::<$ty>()[0];
            dbg!(&res);
            dbg!(&reduced);
            assert!(
                approx::abs_diff_eq!(res, reduced, epsilon = 0.001)
                    || (res.is_nan() && reduced.is_nan())
            );
        };
    }

    rng_test!(u8, 0..0xff, 1_000, u8::wrapping_mul, reduce_prod);
    rng_test!(i8, -128..127, 1_000, i8::wrapping_mul, reduce_prod);
    rng_test!(i64, -128..127, 1_000, i64::wrapping_mul, reduce_prod);
    rng_test!(u64, 0..0xff, 1_000, u64::wrapping_mul, reduce_prod);
    rng_test_float!(
        f32,
        1..1000,
        10,
        |x: f32| (x * 0.01).log2(),
        std::ops::Mul::mul,
        reduce_prod
    );
}

#[test]
fn reduce_and() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();
    use rand::Rng;

    macro_rules! rng_test {
        ($ty:ident, $range:expr, $num:expr, $red:expr, $jit_red:ident) => {
            let mut rng = rand::thread_rng();

            let x = (0..$num)
                .map(|_| rng.gen_range($range))
                .map(|i| i as $ty)
                .collect::<Vec<_>>();
            let reduced = x.to_vec().into_iter().reduce(|a, b| $red(a, b)).unwrap();

            // Launch Kernels:
            let x = tr::array(&x, &device);
            let sum = x.$jit_red();
            sum.schedule();
            let graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>()[0], reduced)
        };
    }

    rng_test!(u8, 0..0x10, 1_000, std::ops::BitAnd::bitand, reduce_and);
    rng_test!(u64, 0..0xff, 1_000, std::ops::BitAnd::bitand, reduce_and);

    let test_bool = |x: &[bool]| {
        // Test bool
        let x = x.to_vec();
        let reduced = x.to_vec().into_iter().reduce(|a, b| a && b).unwrap();

        // Launch Kernels:
        let x = tr::array(&x, &device);
        let res = x.reduce_and();
        res.schedule();
        let graph = tr::compile();
        graph.launch(&device);

        assert_eq!(res.to_vec::<bool>()[0], reduced)
    };

    test_bool(&[true, false, false, false]);
    test_bool(&[false, false, false, true]);
    test_bool(&[false, false, false, false]);
}

#[test]
fn reduce_or() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();
    use rand::Rng;

    macro_rules! rng_test {
        ($ty:ident, $range:expr, $num:expr, $red:expr, $jit_red:ident) => {
            let mut rng = rand::thread_rng();

            let x = (0..$num)
                .map(|_| rng.gen_range($range))
                .map(|i| i as $ty)
                .collect::<Vec<_>>();
            let reduced = x.to_vec().into_iter().reduce(|a, b| $red(a, b)).unwrap();

            // Launch Kernels:
            let x = tr::array(&x, &device);
            let sum = x.$jit_red();
            sum.schedule();
            let graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>()[0], reduced)
        };
    }

    rng_test!(u8, 0..0x10, 1_000, std::ops::BitOr::bitor, reduce_or);
    rng_test!(u64, 0..0xff, 1_000, std::ops::BitOr::bitor, reduce_or);

    let test_bool = |x: &[bool]| {
        // Test bool
        let x = x.to_vec();
        let reduced = x.to_vec().into_iter().reduce(|a, b| a || b).unwrap();

        // Launch Kernels:
        let x = tr::array(&x, &device);
        let res = x.reduce_or();
        res.schedule();
        let graph = tr::compile();
        graph.launch(&device);

        assert_eq!(res.to_vec::<bool>()[0], reduced)
    };

    test_bool(&[true, false, false, false]);
    test_bool(&[false, false, false, true]);
    test_bool(&[false, false, false, false]);
}

#[test]
fn uop_cos() {
    pretty_env_logger::try_init().ok();

    let device = backend::Device::vulkan(0).unwrap();

    let x = [0., 1., std::f32::consts::PI];

    let reference = x.iter().map(|x| x.cos()).collect::<Vec<_>>();

    let x = tr::array(&x, &device);
    let pred = x.cos();
    pred.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    for (reference, pred) in reference.into_iter().zip(pred.to_vec::<f32>().into_iter()) {
        approx::assert_abs_diff_eq!(reference, pred, epsilon = 0.001);
    }
}
