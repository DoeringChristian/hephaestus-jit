use std::fmt::Debug;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use criterion::{BenchmarkId, Throughput};
use hephaestus_jit::backend::vulkan;
use hephaestus_jit::backend::Device;
use hephaestus_jit::tr;

mod benches {
    use super::*;
}

pub fn simple(c: &mut Criterion) {
    c.bench_function("simple", |b| {
        b.iter_batched(
            || {
                let mut a = tr::sized_literal(1, 2);

                for i in 0..10_000 {
                    a = a.add(&tr::literal(i));
                }

                a.schedule();
                a
            },
            |a| {
                let graph = tr::compile();
            },
            BatchSize::PerIteration,
        );
    });
}

criterion_group!(benches, simple);
criterion_main!(benches);
