#[macro_use]
extern crate criterion;
extern crate qrs;
extern crate openblas_src;

mod util;
use util::eig_resolver::{bench_random_10x10_hermitian, bench_random_32x32_hermitian,
                         bench_random_64x64_hermitian};
use criterion::Criterion;
use std::time::Duration;

criterion_group!{   name=benches;
                    config=Criterion::default().sample_size(24)
                        .measurement_time(Duration::new(20, 0));
                    targets=bench_random_10x10_hermitian, bench_random_32x32_hermitian,
                    bench_random_64x64_hermitian
}

criterion_main!(benches);