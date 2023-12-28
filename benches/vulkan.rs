use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hephaestus_jit::backend::vulkan;
use hephaestus_jit::backend::Report;
use hephaestus_jit::tr;

pub fn compress_large() -> std::time::Duration {
    use rand::Rng;

    let device = vulkan(0);

    let n = usize::pow(2, 12);

    // TODO: same bug as in prefix sum but with sizes not divisible by 16
    let src: Vec<bool> = (0..n).map(|_| rand::thread_rng().gen()).collect();

    let src_tr = tr::array(&src, &device);

    let (count, index) = src_tr.compress();

    let mut graph = tr::compile();
    let report = graph.launch(&device);

    let pass = report
        .passes
        .into_iter()
        .find(|pass| pass.name == "Compress Large")
        .unwrap();

    pass.duration
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("compress_large", move |b| {
        b.iter_custom(|iters| {
            let duration = (0..iters)
                .map(|_| compress_large())
                .reduce(|a, b| a + b)
                .unwrap()
                .div_f64(iters as f64);
            duration
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
