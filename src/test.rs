use half::f16;
use rand::Rng;
use std::collections::HashSet;
use std::sync::Mutex;
use std::thread;

use approx::assert_abs_diff_eq;

use crate::record::{self, *};
use crate::tr::VarRef;
use crate::vartype::{self, AsVarType, Instance, Intersection};
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
    tr::TS.with(|s| {
        dbg!(&s);
    });

    let mut graph = tr::compile();
    dbg!(&graph);
    for i in 0..1000 {
        graph.launch(&device);
    }

    dbg!(graph.n_passes());

    dbg!(i.to_vec::<u32>(..));
    dbg!(j.to_vec::<u32>(..));
    assert_eq!(i.to_vec::<u32>(..), vec![1, 2, 3, 4, 5, 5, 6, 7, 8, 9]);
    assert_eq!(j.to_vec::<u32>(..), vec![0, 1, 2, 3, 4]);
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
    tr::TS.with(|s| {
        dbg!(&s);
    });
    let mut graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    dbg!(c.to_vec::<u16>(..));
    assert_eq!(c.to_vec::<u16>(..), vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
}
#[test]
fn simple_f16() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let c = tr::index(10).cast(f16::var_ty());

    c.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    let reference = (0..10).map(|i| f16::from_f32(i as _)).collect::<Vec<_>>();
    assert_eq!(reference, c.to_vec::<f16>(..));
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
    tr::TS.with(|s| {
        dbg!(&s);
    });

    let mut graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    // dbg!(&b1.data().buffer().unwrap().to_host::<i32>().unwrap());
    assert_eq!(b1.to_vec::<i32>(..), vec![2, 2, 2, 2, 2]);
}
#[test]
fn scatter_chain2() {
    let device = backend::Device::vulkan(0).unwrap();

    let a = tr::sized_literal(0, 5);
    let b = a.add(&tr::literal(1));
    tr::literal(1).scatter(&a, &tr::index(5));

    b.schedule();

    let mut graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    dbg!(b.to_vec::<i32>(..));
    dbg!(a.to_vec::<i32>(..));
    assert_eq!(b.to_vec::<i32>(..), vec![2, 2, 2, 2, 2]);
    assert_eq!(a.to_vec::<i32>(..), vec![1, 1, 1, 1, 1]);
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

    dbg!(v.to_vec::<f32>(..));
    dbg!(a.to_vec::<f32>(..));
    dbg!(b.to_vec::<f32>(..));
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

    dbg!(&s.to_vec::<u8>(..));
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

    dbg!(s.to_vec::<u8>(..));
    dbg!(a.to_vec::<u8>(..));
    dbg!(b.to_vec::<u32>(..));
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
    dbg!(v.to_vec::<f32>(..));
    assert_eq!(v.to_vec::<f32>(..), vec![1.0; 8]);
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
    dbg!(v.to_vec::<f32>(..));
    assert_eq!(v.to_vec::<f32>(..), vec![1.0; 8]);
}
#[test]
fn conditionals() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let dst = tr::sized_literal(true, 100);

    dst.schedule();

    let mut graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    dbg!(&dst.to_vec::<u8>(..));
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
    dbg!(&active.to_vec::<u8>(..));

    tr::literal(1).scatter_if(&dst, &tr::index(10), &active);

    dst.schedule();

    let mut graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    assert_eq!(dst.to_vec::<i32>(..), vec![1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
}
#[test]
fn conditional_gather() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let src = tr::sized_literal(1, 10);
    let active = tr::array(
        &[
            true, true, false, false, true, false, true, false, true, false,
        ],
        &device,
    );

    let dst = src.gather_if(&tr::index(10), &active);
    dst.schedule();

    let mut graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    assert_eq!(dst.to_vec::<i32>(..), vec![1, 1, 0, 0, 1, 0, 1, 0, 1, 0]);
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

    let mut graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    assert_eq!(res.to_vec::<i32>(..), vec![10, 5]);
}
#[test]
fn accel() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let x = tr::array(&[1f32, 0f32, 1f32], &device);
    let y = tr::array(&[0f32, 1f32, 1f32], &device);
    let z = tr::array(&[0f32, 0f32, 0f32], &device);
    let vertices = tr::vec(&[&x, &y, &z]);

    let triangles = tr::array(&[[0u32, 1u32, 2u32]], &device);

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

    dbg!(instances.to_vec::<u32>(..));

    let desc = tr::AccelDesc {
        geometries: &[tr::GeometryDesc::Triangles {
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
    tr::TS.with(|s| {
        let s = s.borrow();
        dbg!(&s);
        for v in s.scheduled.values() {
            tr::with_trace(|trace| {
                dbg!(trace.var(v.id()));
            })
        }
    });

    let mut graph = tr::compile();
    graph.launch(&device);

    dbg!(intersection.to_vec::<i32>(..));
    let intersections = intersection.to_vec::<Intersection>(..);

    assert_abs_diff_eq!(intersections[0].bx, 0.4, epsilon = 0.001);
    assert_abs_diff_eq!(intersections[0].by, 0.2, epsilon = 0.001);
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
            let mut graph = tr::compile();
            graph.launch(&device);

            assert_eq!(max.to_vec::<$ty>(..)[0], reduced)
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
            let mut graph = tr::compile();
            graph.launch(&device);

            assert_eq!(min.to_vec::<$ty>(..)[0], reduced)
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
            let mut graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>(..)[0], reduced)
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
            let mut graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>(..)[0], reduced)
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
            let mut graph = tr::compile();
            graph.launch(&device);

            let res = sum.to_vec::<$ty>(..)[0];
            dbg!(&res);
            dbg!(&reduced);
            assert!(
                approx::abs_diff_eq!(res, reduced, epsilon = 0.01)
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
            let mut graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>(..)[0], reduced)
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
        let mut graph = tr::compile();
        graph.launch(&device);

        assert_eq!(res.to_vec::<bool>(..)[0], reduced)
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
            let mut graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>(..)[0], reduced)
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
        let mut graph = tr::compile();
        graph.launch(&device);

        assert_eq!(res.to_vec::<bool>(..)[0], reduced)
    };

    test_bool(&[true, false, false, false]);
    test_bool(&[false, false, false, true]);
    test_bool(&[false, false, false, false]);
}
#[test]
fn reduce_xor() {
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
            let mut graph = tr::compile();
            graph.launch(&device);

            assert_eq!(sum.to_vec::<$ty>(..)[0], reduced)
        };
    }

    rng_test!(u8, 0..0x10, 1_000, std::ops::BitXor::bitxor, reduce_xor);
    rng_test!(u64, 0..0xff, 1_000, std::ops::BitXor::bitxor, reduce_xor);

    let test_bool = |x: &[bool]| {
        // Test bool
        let x = x.to_vec();
        let reduced = x.to_vec().into_iter().reduce(|a, b| a || b).unwrap();

        // Launch Kernels:
        let x = tr::array(&x, &device);
        let res = x.reduce_or();
        res.schedule();
        let mut graph = tr::compile();
        graph.launch(&device);

        assert_eq!(res.to_vec::<bool>(..)[0], reduced)
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

    let mut graph = tr::compile();
    graph.launch(&device);

    for (reference, pred) in reference
        .into_iter()
        .zip(pred.to_vec::<f32>(..).into_iter())
    {
        approx::assert_abs_diff_eq!(reference, pred, epsilon = 0.001);
    }
}
#[test]
fn scatter_atomic() {
    pretty_env_logger::try_init().ok();

    let device = backend::Device::vulkan(0).unwrap();

    let src = tr::literal(1u32);

    let dst = tr::array(&[0u32, 0, 0], &device);

    let n = 16;
    let idx = tr::sized_literal(0, n);

    let prev = src.scatter_atomic(&dst, &idx, crate::op::ReduceOp::Sum);

    // NOTE: in contrast to scatter and scatter_reduce, atomic operations require scheduling of the
    // result as it could lead to unintended evaluation of the result.
    prev.schedule();

    let mut graph = tr::compile();
    graph.launch(&device);

    assert_eq!(dst.to_vec::<u32>(..)[0], n as u32);

    let mut uniq = HashSet::new();
    assert!(
        prev.to_vec::<u32>(..)
            .into_iter()
            .all(move |x| uniq.insert(x)),
        "Atomic Operations should return the previous index which is unique!"
    );
}

#[test]
fn scatter_reduce() {
    pretty_env_logger::try_init().ok();

    let device = backend::Device::vulkan(0).unwrap();

    let src = tr::literal(1u32);

    let dst = tr::array(&[0u32, 0, 0], &device);

    let n = 16;
    let idx = tr::sized_literal(0, n);

    src.scatter_reduce(&dst, &idx, crate::op::ReduceOp::Sum);

    let mut graph = tr::compile();
    graph.launch(&device);

    assert_eq!(dst.to_vec::<u32>(..)[0], n as u32);
}

#[test]
fn compress_small() {
    use rand::Rng;

    pretty_env_logger::try_init().ok();

    let device = backend::Device::vulkan(0).unwrap();

    let src: Vec<bool> = (0..128).map(|_| rand::thread_rng().gen()).collect();

    let reference = src
        .iter()
        .enumerate()
        .filter(|(_, b)| **b)
        .map(|(i, _)| i as u32)
        .collect::<Vec<_>>();

    let src_tr = tr::array(&src, &device);

    let (count, index) = src_tr.compress();

    let mut graph = tr::compile();
    graph.launch(&device);

    let count = count.to_vec::<u32>(..)[0] as usize;
    let mut prediction = index.to_vec::<u32>(..);
    prediction.truncate(count);

    assert_eq!(reference, prediction);
    assert_eq!(reference.len(), count);
}
#[test]
fn compress_large() {
    use rand::Rng;

    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let n = usize::pow(2, 12) + 15;

    // TODO: same bug as in prefix sum but with sizes not divisible by 16
    // let src: Vec<bool> = (0..n).map(|_| rand::thread_rng().gen()).collect();
    let src = (0..n).map(|_| true).collect::<Vec<_>>();

    let reference = src
        .iter()
        .enumerate()
        .filter(|(_, b)| **b)
        .map(|(i, _)| i as u32)
        .collect::<Vec<_>>();

    let src_tr = tr::array(&src, &device);

    let (count, index) = src_tr.compress();

    let mut graph = tr::compile();
    graph.launch(&device);

    let count = count.to_vec::<u32>(..)[0] as usize;
    let mut prediction = index.to_vec::<u32>(..);
    prediction.truncate(count);

    assert_eq!(reference, prediction);
    assert_eq!(reference.len(), count);
}
#[test]
fn prefix_sum() {
    pretty_env_logger::try_init().ok();

    // TODO: investigate why it's not working with sizes not divisible by 4 (suspect last thread
    // not running)
    let num = 2048 * 4 + 3; // test some weird value for initialization

    let input = (0..num as u64).map(|i| i).collect::<Vec<_>>();

    let device = vulkan(0);

    let x = tr::array(&input, &device);

    let prediction = x.prefix_sum(true);
    prediction.schedule();

    let mut graph = tr::compile();
    graph.launch(&device);

    let reference = input
        .into_iter()
        .scan(0, |sum, i| {
            *sum += i;
            Some(*sum)
        })
        .collect::<Vec<_>>();

    assert_eq!(prediction.to_vec::<u64>(..), reference);
}
#[test]
fn dynamic_index() {
    use rand::Rng;
    pretty_env_logger::try_init().ok();

    // TODO: figure out compression bug
    // Also, add a view function, that allows for a view into a subset of a buffer without
    // having to call [`tr::gather`].
    let n = 1024;
    let min = 3;
    let max = 7;

    let device = vulkan(0);

    // TODO: same bug as in prefix sum but with sizes not divisible by 16
    let src: Vec<i32> = (0..n)
        .map(|_| rand::thread_rng().gen_range(0..10))
        .collect();

    let src_var = tr::array(&src, &device);

    // Compress
    let cond = src_var.lt(&tr::literal(max));
    let indices = cond.compress_dyn();
    let values = src_var.gather(&indices);

    let cond = values.gt(&tr::literal(min));
    let indices = cond.compress_dyn();
    let values = values.gather(&indices);

    values.schedule();

    let mut graph = tr::compile();
    graph.launch(&device);

    assert_eq!(values.capacity(), n);

    let values = values.to_vec::<i32>(..);

    let reference = src
        .into_iter()
        .filter(|i| *i > min && *i < max)
        .collect::<Vec<_>>();

    assert_eq!(values, reference);
}
#[test]
fn example() {
    use rand::Rng;

    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let n = 128;

    // Create random values and mask
    let a = tr::array(
        &(0..n)
            .map(|_| rand::thread_rng().gen_range(0f32..1f32))
            .collect::<Vec<_>>(),
        &device,
    );
    let mask = tr::sized_literal(true, n);

    let mut f = tr::record(|()| {
        // Compress wavefront
        let indices = mask.compress_dyn();
        let b = a.gather(&indices);

        // Do some (RR style) work on the values
        let b = b.mul(&tr::literal(0.9f32));
        let new_mask = b.gt(&tr::literal(0.1f32));

        // Write wavefront back to arrays
        new_mask.scatter(&mask, &indices);
        b.scatter(&a, &indices);

        a.schedule();
    });

    // Launch it multiple times
    for _ in 0..10 {
        f(&device, ());
    }

    // Read data back to CPU and print it
    dbg!(a.to_vec::<f32>(..));
}
#[test]
fn record() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let mut f = tr::record(|a: VarRef| {
        a.add(&tr::literal(1)).scatter(&a, &tr::index(3));
    });

    let a = tr::array(&[1, 2, 3], &device);

    f(&device, a.clone());
    assert_eq!(a.to_vec::<i32>(..), vec![2, 3, 4]);

    let b = tr::array(&[4, 5, 6], &device);

    f(&device, b.clone());
    dbg!(&a);
    dbg!(&b);
    assert_eq!(a.to_vec::<i32>(..), vec![2, 3, 4]);
    assert_eq!(b.to_vec::<i32>(..), vec![5, 6, 7]);
}
#[test]
fn matrix_times_matrix() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let c0 = tr::vec(&[&tr::sized_literal(1f32, 1), &tr::literal(3f32)]);
    let c1 = tr::vec(&[&tr::literal(2f32), &tr::literal(4f32)]);
    let m0 = tr::mat(&[&c0, &c1]);

    let c0 = tr::vec(&[&tr::literal(5f32), &tr::literal(7f32)]);
    let c1 = tr::vec(&[&tr::literal(6f32), &tr::literal(8f32)]);
    let m1 = tr::mat(&[&c0, &c1]);

    let res = m0.mul(&m1);
    res.schedule();

    let graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    dbg!(res.to_vec::<f32>(..));
}
#[test]
fn array() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let a0 = tr::sized_literal(1, 2);
    let a1 = tr::literal(2);
    let a2 = tr::literal(3);

    let array = tr::arr(&[&a0, &a1, &a2]);
    array.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    assert_eq!(array.to_vec::<[i32; 3]>(..), vec![[1, 2, 3], [1, 2, 3]]);
}
#[test]
fn dyn_extract() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let a0 = tr::sized_literal(1, 2);
    let a1 = tr::literal(2);
    let a2 = tr::literal(3);

    let array = tr::arr(&[&a0, &a1, &a2]);

    let idx = tr::index(2);

    let res = array.extract_dyn(&idx);
    res.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    dbg!(res.to_vec::<i32>(..));
}
#[test]
fn vec3_memory_layout() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let vec = tr::vec(&[&tr::sized_literal(1, 2), &tr::literal(2), &tr::literal(3)]);

    let tmp = vec.gather(&tr::index(2));

    vec.schedule();
    tmp.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    assert_eq!(vec.to_vec::<i32>(..), vec![1, 2, 3, 1, 2, 3]);
}
#[test]
fn cast_array_vec() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let arr = tr::arr(&[
        &tr::sized_literal(1f32, 2),
        &tr::literal(2f32),
        &tr::literal(3f32),
    ]);
    let vec = arr.cast(vartype::vector(f32::var_ty(), 3));
    let arr = vec.cast(vartype::array(i32::var_ty(), 3));

    vec.schedule();
    arr.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    // assert_eq!(vec.to_vec::<i32>(..), vec![1, 2, 3, 1, 2, 3]);
    assert_eq!(arr.to_vec::<i32>(..), vec![1, 2, 3, 1, 2, 3]);
}
#[test]
fn atomic_inc() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let atomics = tr::array(&[0u32, 0, 0], &device);

    let active = tr::sized_literal(true, 1000);
    let ids = atomics.atomic_inc(&tr::literal(1u32), &active);

    ids.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    // dbg!(ids.to_vec::<u32>(..));

    let mut uniq = HashSet::new();
    assert!(
        ids.to_vec::<u32>(..)
            .into_iter()
            .all(move |x| uniq.insert(x)),
        "Atomic Operations should return the previous index which is unique!"
    );
}
#[test]
fn atomic_inc_rand() {
    pretty_env_logger::try_init().ok();

    let device = vulkan(0);

    let atomics = tr::array(&[0u32, 0, 0], &device);

    let mut rng = rand::thread_rng();
    let active_vec = (0..1000).map(|_| rng.gen_bool(0.5)).collect::<Vec<_>>();
    let active = tr::array(&active_vec, &device);

    let ids = atomics.atomic_inc(&tr::literal(1u32), &active);

    let (idxs, len) = active.compress();

    let uids = ids.gather(&idxs);

    uids.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    // dbg!(ids.to_vec::<u32>(..));

    let mut uniq = HashSet::new();
    assert!(
        uids.to_vec::<u32>(..)
            .into_iter()
            .all(move |x| uniq.insert(x)),
        "Atomic Operations should return the previous index which is unique!"
    );
}

#[test]
fn scope() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let x = tr::sized_literal(1, 2);

    let ym = Mutex::new(None);
    let yr = &ym;

    thread::scope(|s| {
        let thread = s.spawn(move || {
            let _scope = tr::new_scope();
            let y = tr::literal(1);
            *yr.lock().unwrap() = Some(y);
        });

        thread.join().unwrap();
    });

    let y = std::mem::take(&mut (*ym.lock().unwrap())).unwrap();

    let z = x.add(&y);

    z.schedule();

    let graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    assert_eq!(y.scope(), z.scope());
    assert!(x.scope() < y.scope());

    assert_eq!(z.to_vec::<i32>(..), vec![2, 2]);
}
#[test]
fn loop_record1() {
    // TODO: loops work without scopes -> comment out scopes
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    // Initial state
    let i = tr::array(&[0, 1], &device);
    let c = tr::literal(true);

    // Start the loop recording
    let (loop_start, state) = tr::loop_start(&[&c, &i]);
    let c = state[0].clone();
    let i = state[1].clone();

    let i = i.add(&tr::literal(1));
    let c = c.and(&i.lt(&tr::literal(2)));

    let state = tr::loop_end(&loop_start, &[&c, &i]);
    let c = state[0].clone();
    let i = state[1].clone();

    i.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    assert_eq!(i.to_vec::<i32>(..), vec![2, 2]);
}

#[test]
fn loop_record2() {
    // TODO: loops work without scopes -> comment out scopes
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    // Initial state
    let mut i = tr::array(&[0, 1], &device);
    let mut c = tr::literal(true);

    loop_record!([i] while c {
        i = i.add(&tr::literal(1));
        c = c.and(&i.lt(&tr::literal(2)));
    });

    i.schedule();
    c.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    assert_eq!(i.to_vec::<i32>(..), vec![2, 2]);
    assert_eq!(c.to_vec::<bool>(..), vec![false, false]);
}
#[test]
#[allow(non_snake_case)]
fn matmul_linspace() {
    let N = 128;
    let M = 128;
    let K = 128;
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    pub fn linspace(start: f16, end: f16, num: usize) -> VarRef {
        tr::literal(start).add(
            &tr::index(num)
                .cast(f16::var_ty())
                .mul(&tr::literal((end - start) / (f16::from_f32(num as _)))),
        )
    }

    let A = linspace(f16::from_f32(0f32), f16::from_f32(1f32), M * K);
    let B = linspace(f16::from_f32(0f32), f16::from_f32(1f32), K * N);
    let C = tr::sized_literal(f16::ZERO, M * N);

    let D = tr::matmul(&A, &B, &C, M, N, K);

    A.schedule();
    B.schedule();
    D.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    let D_ref = {
        use ndarray::prelude::*;
        let A = Array::linspace(0f32, 1f32, M * K);
        let B = Array::linspace(0f32, 1f32, K * N);
        let A = A.into_shape([M, K]).unwrap();
        let B = B.into_shape([K, N]).unwrap();

        let C = A.dot(&B);

        let C = C.into_raw_vec();
        let C = C.into_iter().map(|e| f16::from_f32(e)).collect::<Vec<_>>();
        C
    };

    // dbg!(A.to_vec::<f16>(..));
    // dbg!(B.to_vec::<f16>(..));
    // dbg!(C.to_vec::<f16>(..));
    // dbg!(&C_ref);
    use num_traits::*;
    let D = D.to_vec::<f16>(..);
    assert!(
        D.iter()
            .zip(&D_ref)
            .all(|(&D, &D_ref)| (D - D_ref).abs() < f16::from_f32(1.0)),
        "lhs = {D:?}\n is not equal to rhs = {D_ref:?}\n"
    );
}
