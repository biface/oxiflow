//! Criterion benchmark entry point (#53).
//!
//! A single binary (`oxiflow_benchmarks`, matching the `[[bench]]` slot in
//! `Cargo.toml`) combining every model's benchmarks via `mod`, rather than
//! one `[[bench]]` target per model — avoids a `Cargo.toml` edit each time
//! #137/#138/#139 add a new model.

mod diffusion_1d;
mod langmuir_multi;
mod viscous_burgers;

use criterion::{criterion_group, criterion_main};

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    diffusion_1d::full_run,
    diffusion_1d::laplacian_dispatch,
    diffusion_1d::raw_rayon_dispatch_diagnostic,
    viscous_burgers::full_run,
    viscous_burgers::combined_dispatch_diagnostic,
    langmuir_multi::full_run,
    langmuir_multi::row_dispatch_diagnostic
);
#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    diffusion_1d::full_run,
    viscous_burgers::full_run,
    langmuir_multi::full_run
);

criterion_main!(benches);
