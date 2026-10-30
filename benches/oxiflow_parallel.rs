//! Criterion benchmark entry point for parallel-dispatch calibration
//! ([#53](https://github.com/biface/oxiflow/issues/53), [DA-048](https://github.com/biface/oxiflow/issues/140)).
//!
//! A single binary (`oxiflow_parallel`, matching the `[[bench]]` slot in
//! `Cargo.toml` -- named for its object, since other `[[bench]]` targets
//! for other purposes are expected later) combining every model's
//! benchmarks via `mod`, rather than one `[[bench]]` target per model —
//! avoids a `Cargo.toml` edit each time a new model joins the suite.

// Shared models used by more than one benchmark module, included once here
// (a file loaded as two modules gives two sets of distinct types, and
// `clippy::duplicate_mod` rejects it); the modules reach them through
// `crate::`.
#[path = "../rsc/models/lahar_flux.rs"]
mod lahar_flux;
#[cfg(feature = "parallel")]
#[path = "../rsc/models/lahar_outfall.rs"]
mod lahar_outfall;
#[path = "../rsc/models/lahar_physics.rs"]
mod lahar_physics;
#[path = "../rsc/models/lahar_source.rs"]
mod lahar_source;
#[path = "../rsc/models/lahar_validation.rs"]
mod lahar_validation;
#[path = "../rsc/models/langmuir_multi.rs"]
mod langmuir_multi_model;

mod diffusion_1d;
#[cfg(feature = "parallel")]
mod heat_transfer_1d;
#[cfg(feature = "parallel")]
mod implicit_dispatch;
mod lahar_multifield;
mod langmuir_multi;
#[cfg(feature = "parallel")]
mod orchestrator_phase1;
mod viscous_burgers;

use criterion::{criterion_group, criterion_main};

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    diffusion_1d::full_run,
    diffusion_1d::laplacian_dispatch,
    diffusion_1d::raw_rayon_dispatch_diagnostic,
    diffusion_1d::implicit_jacobian_dispatch,
    diffusion_1d::boundary_condition_overhead,
    heat_transfer_1d::coefficient_dispatch,
    viscous_burgers::full_run,
    viscous_burgers::combined_dispatch_diagnostic,
    langmuir_multi::full_run,
    langmuir_multi::row_dispatch_diagnostic,
    langmuir_multi::implicit_jacobian_dispatch,
    lahar_multifield::full_run,
    lahar_multifield::source_dispatch_diagnostic,
    lahar_multifield::flux_dispatch_diagnostic,
    orchestrator_phase1::langmuir_multi_domains,
    orchestrator_phase1::lahar_multifield_domains,
    orchestrator_phase1::lahar_lake_wave_domains
);
#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    diffusion_1d::full_run,
    diffusion_1d::boundary_condition_overhead,
    viscous_burgers::full_run,
    langmuir_multi::full_run,
    lahar_multifield::full_run
);

criterion_main!(benches);
