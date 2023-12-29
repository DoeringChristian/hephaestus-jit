use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::{BenchmarkId, Throughput};
use hephaestus_jit::backend::vulkan;
use hephaestus_jit::backend::Device;
use hephaestus_jit::tr;

pub fn compress_large(device: &Device, n: usize) -> std::time::Duration {
    // let src_tr = tr::array(&src, &device);
    let src_tr = tr::sized_literal(true, n);

    let (count, index) = src_tr.compress();

    let mut graph = tr::compile();
    let report = graph.launch(&device);

    assert_eq!(count.to_vec::<u32>()[0], n as u32);

    let pass = report
        .passes
        .into_iter()
        .find(|pass| pass.name == "Compress Large")
        .unwrap();

    pass.duration
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_large");

    let device = vulkan(0);

    for i in 10..30 {
        let n = usize::pow(2, i);

        group.throughput(Throughput::Elements(n as _));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, n| {
            b.iter_custom(|iters| {
                let duration = (0..iters)
                    .map(|_| {
                        assert!(tr::is_empty());
                        compress_large(&device, *n)
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
