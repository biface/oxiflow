//! Criterion benchmark entry point for parallel-dispatch calibration
//! (#53, DD-048).
//!
//! A single binary (`oxiflow_parallel`, matching the `[[bench]]` slot in
//! `Cargo.toml` -- named for its object, since other `[[bench]]` targets
//! for other purposes are expected later) combining every model's
//! benchmarks via `mod`, rather than one `[[bench]]` target per model —
//! avoids a `Cargo.toml` edit each time a new model joins the suite.

mod diffusion_1d;
mod lahar_multifield;
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
    langmuir_multi::row_dispatch_diagnostic,
    lahar_multifield::full_run,
    lahar_multifield::source_dispatch_diagnostic,
    lahar_multifield::flux_dispatch_diagnostic
);
#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    diffusion_1d::full_run,
    viscous_burgers::full_run,
    langmuir_multi::full_run,
    lahar_multifield::full_run
);

criterion_main!(benches);
