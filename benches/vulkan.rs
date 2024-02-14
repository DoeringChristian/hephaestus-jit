use std::fmt::Debug;

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration,
};
use criterion::{BenchmarkId, Throughput};
use hephaestus_jit::backend::vulkan;
use hephaestus_jit::backend::Device;
use hephaestus_jit::tr;

mod benches {
    use half::f16;
    use hephaestus_jit::tr::VarRef;
    use hephaestus_jit::vartype::AsVarType;

    use super::*;
    #[allow(non_snake_case)]
    pub fn cooperative_matrix(
        device: &Device,
        n: usize,
        m: usize,
        k: usize,
    ) -> std::time::Duration {
        pub fn linspace(start: f16, end: f16, num: usize) -> VarRef {
            tr::literal(start).add(
                &tr::index(num)
                    .cast(f16::var_ty())
                    .mul(&tr::literal((end - start) / (f16::from_f32(num as _)))),
            )
        }
        let A = linspace(f16::from_f32(0f32), f16::from_f32(1f32), n * k);
        let B = linspace(f16::from_f32(0f32), f16::from_f32(1f32), k * m);

        let C = tr::matmul(&A, &B, n, m, k);

        C.schedule();

        let graph = tr::compile();
        let report = graph.launch(&device);

        let pass = report
            .passes
            .into_iter()
            .find(|pass| pass.name == "Cooperative Matrix Multiply")
            .unwrap();

        pass.duration
    }
    pub fn compress_large(device: &Device, n: usize) -> std::time::Duration {
        let src_tr = tr::sized_literal(true, n);

        let (count, index) = src_tr.compress();

        let graph = tr::compile();
        let report = graph.launch(&device);

        assert_eq!(count.item::<u32>(), n as u32);

        let pass = report
            .passes
            .into_iter()
            .find(|pass| pass.name == "Compress Large")
            .unwrap();

        pass.duration
    }
    pub fn prefix_sum_large<T>(device: &Device, n: usize, init: T, sum: T) -> std::time::Duration
    where
        T: hephaestus_jit::vartype::AsVarType + Eq + Debug,
    {
        let src = tr::sized_literal(init, n);

        let pfs = src.prefix_sum(false);

        let graph = tr::compile();
        let report = graph.launch(&device);

        assert_eq!(pfs.to_vec::<T>(n - 1..n)[0], sum);

        let pass = report
            .passes
            .into_iter()
            .find(|pass| pass.name == "Prefix Sum Large")
            .unwrap();

        pass.duration
    }
}

pub fn measure_custom(
    c: &mut Criterion,
    range: std::ops::Range<u32>,
    name: &str,
    f: impl Fn(usize) -> std::time::Duration,
) {
    let mut group = c.benchmark_group(name);
    for i in range {
        let n = usize::pow(2, i);

        group.throughput(Throughput::Elements(n as _));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, n| {
            b.iter_custom(|iters| {
                let duration = (0..iters)
                    .map(|_| {
                        assert!(tr::is_empty());
                        black_box(f(*n))
                    })
                    .reduce(|a, b| a + b)
                    .unwrap();
                duration
            })
        });
    }
    group.finish();
}

pub fn compress_large(c: &mut Criterion) {
    let device = vulkan(0);

    measure_custom(c, 10..30, "compress_large", |n| {
        benches::compress_large(&device, n)
    });
}
pub fn prefix_sum_large(c: &mut Criterion) {
    let device = vulkan(0);

    measure_custom(c, 10..30, "prefix_sum_large_u32", |n| {
        benches::prefix_sum_large(&device, n, 1u32, n as u32)
    });
}
pub fn cooperative_matrix(c: &mut Criterion) {
    let device = vulkan(0);

    let mut group = c.benchmark_group("cooperative_matrix_f16");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for i in 4..=12 {
        let n = usize::pow(2, i);
        let m = n;
        let k = n;

        group.throughput(Throughput::Elements((n * m * k) as _));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_custom(|iters| {
                let duration = (0..iters)
                    .map(|_| {
                        assert!(tr::is_empty());
                        black_box(benches::cooperative_matrix(&device, n, m, k))
                    })
                    .reduce(|a, b| a + b)
                    .unwrap();
                duration
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    compress_large,
    prefix_sum_large,
    cooperative_matrix
);
criterion_main!(benches);
