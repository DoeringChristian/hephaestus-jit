use std::fmt::Debug;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::{BenchmarkId, Throughput};
use hephaestus_jit::backend::vulkan;
use hephaestus_jit::backend::Device;
use hephaestus_jit::tr;

mod benches {
    use super::*;
}

pub fn simple(c: &mut Criterion) {
    let device = vulkan(0);

    let mut a = tr::sized_literal(1, 2);

    for i in 0..100 {
        a = a.add(&tr::literal(i));
    }

    c.bench_function("simple", |b| {
        b.iter(|| {
            a.schedule();

            let graph = tr::compile();
        })
    });
}

criterion_group!(benches, simple);
criterion_main!(benches);
