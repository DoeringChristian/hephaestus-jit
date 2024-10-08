use std::fmt::Debug;

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration,
};
use criterion::{BenchmarkId, Throughput};
use hephaestus_jit::backend::vulkan;
use hephaestus_jit::backend::Device;
use hephaestus_jit::tr;
use num_traits::Pow;

mod benches {
    use std::time::{Duration, Instant};

    use half::f16;
    use hephaestus_jit::tr::VarRef;
    use hephaestus_jit::vartype::AsVarType;

    use super::*;

    pub fn fused_mlp(device: &Device, batch_size: usize) -> std::time::Duration {
        let width = 64;
        let in_width = width;
        let out_width = width;
        let hidden_layers = 2;

        let input = tr::sized_literal(f16::from_f32(1f32), batch_size * in_width);
        let weights = tr::sized_literal(f16::from_f32(1f32), width * width * (2 + hidden_layers));

        input.schedule();
        weights.schedule();
        tr::compile().unwrap().launch(&device);

        let output = tr::fused_mlp_inference(
            &input,
            &weights,
            width,
            in_width,
            out_width,
            hidden_layers,
            batch_size,
        );
        output.schedule();

        let graph = tr::compile();

        let report = graph.unwrap().launch(&device).unwrap();

        // let pass = report
        //     .backend
        //     .exec
        //     .unwrap()
        //     .passes
        //     .into_iter()
        //     .find(|pass| pass.name == "Fused MLP")
        //     .unwrap();

        // pass.duration
        report.backend.exec.cpu_duration
    }

    #[allow(non_snake_case)]
    pub fn cooperative_matrix(
        device: &Device,
        m: usize,
        n: usize,
        k: usize,
    ) -> std::time::Duration {
        pub fn linspace(start: f32, end: f32, num: usize) -> VarRef {
            tr::literal(start).add(
                &tr::sized_index(num)
                    .cast(f32::var_ty())
                    .mul(&tr::literal((end - start) / (num as f32))),
            )
        }

        let A = linspace(0f32, 1f32, m * k).cast(f16::var_ty());
        let B = linspace(0f32, 1f32, k * n).cast(f16::var_ty());
        let C = tr::sized_literal(f16::from_f32(0f32), n * m);

        A.schedule();
        B.schedule();
        C.schedule();

        let graph = tr::compile().unwrap();
        graph.launch(&device);

        let C = tr::matfma(&A, &B, &C, m, n, k);

        C.schedule();

        let graph = tr::compile().unwrap();
        let report = graph.launch(&device).unwrap();

        let pass = report
            .backend
            .exec
            .passes
            .into_iter()
            .find(|pass| pass.name == "Cooperative Matrix Multiply")
            .unwrap();

        // report.cpu_time
        pass.duration
    }
    pub fn compress_large(device: &Device, n: usize) -> std::time::Duration {
        let src_tr = tr::sized_literal(true, n);

        let (count, index) = src_tr.compress();

        let graph = tr::compile().unwrap();
        let report = graph.launch(&device).unwrap();

        assert_eq!(count.item::<u32>(), n as u32);

        let pass = report
            .backend
            .exec
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

        let graph = tr::compile().unwrap();
        let report = graph.launch(&device).unwrap();

        assert_eq!(pfs.to_vec::<T>(n - 1..n)[0], sum);

        let pass = report
            .backend
            .exec
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
    for i in 7..=12 {
        let m = usize::pow(2, i);
        let n = m;
        let k = m;

        // NOTE: A matrix-matrix multiplication costs m * n * k * 2 flops
        group.throughput(Throughput::Elements((m * n * k * 2) as _));

        group.bench_with_input(BenchmarkId::from_parameter(m), &n, |b, &n| {
            b.iter_custom(|iters| {
                let duration = (0..iters)
                    .map(|_| {
                        assert!(tr::is_empty());
                        black_box(benches::cooperative_matrix(&device, m, n, k))
                    })
                    .reduce(|a, b| a + b)
                    .unwrap();
                duration
            })
        });
    }
    group.finish();
}
pub fn fused_mlp(c: &mut Criterion) {
    #[cfg(feature = "profile-with-puffin")]
    let _puffin_server = {
        let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
        let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
        puffin::set_scopes_on(true);
        puffin_server
    };

    let device = vulkan(0);

    let mut group = c.benchmark_group("fused_mlp");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for i in 14..=21 {
        let batch_size = 2usize.pow(i);

        group.throughput(Throughput::Elements(batch_size as _));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter_custom(|iters| {
                    let duration = (0..iters)
                        .map(|_| {
                            assert!(tr::is_empty());
                            let duration = black_box(benches::fused_mlp(&device, batch_size));
                            profiling::finish_frame!();
                            duration
                        })
                        .reduce(|a, b| a + b)
                        .unwrap();
                    duration
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    compress_large,
    prefix_sum_large,
    cooperative_matrix,
    fused_mlp,
);
criterion_main!(benches);
