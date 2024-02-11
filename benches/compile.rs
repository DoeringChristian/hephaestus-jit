use std::fmt::Debug;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hephaestus_jit::tr;

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
