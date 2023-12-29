use std::fmt::Debug;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::{BenchmarkId, Throughput};
use hephaestus_jit::backend::vulkan;
use hephaestus_jit::backend::Device;
use hephaestus_jit::tr;

mod benches {
    use super::*;
    pub fn compress_large(device: &Device, n: usize) -> std::time::Duration {
        let src_tr = tr::sized_literal(true, n);

        let (count, index) = src_tr.compress();

        let mut graph = tr::compile();
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

        let mut graph = tr::compile();
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
                    .unwrap()
                    .div_f64(iters as f64);
                duration
            })
        });
    }
    group.finish();
}

pub fn compress_large(c: &mut Criterion) {
    let device = vulkan(0);

    measure_custom(c, 10..31, "compress_large", |n| {
        benches::compress_large(&device, n)
    });
}
pub fn prefix_sum_large(c: &mut Criterion) {
    let device = vulkan(0);

    measure_custom(c, 10..31, "prefix_sum_large_u32", |n| {
        benches::prefix_sum_large(&device, n, 1u32, n as u32)
    });
}

criterion_group!(benches, compress_large, prefix_sum_large);
criterion_main!(benches);
