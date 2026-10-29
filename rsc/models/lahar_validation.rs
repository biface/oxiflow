//! Lahar validation scenario of paper Sec. 4.1 (flat, non-erodible bed, pure
//! water released at a constant rate): the source term of
//! `rsc/models/lahar_source.rs` composed with the shallow-layer flux of
//! `rsc/models/lahar_flux.rs`, the bed-slope calculator the source reads, and
//! the front position the runs are compared on.
//!
//! Single definition used by `examples/lahar_multifield.rs`,
//! `examples/lahar_multifield_plot.rs`, `tests/lahar_multifield_analytical.rs`
//! and `benches/lahar_multifield.rs`, each of which includes this file as a
//! module with `#[path]`. The drag regime, the node count and the half-width
//! of the domain `[-x_half_width, x_half_width]` are parameters of every
//! builder; the run settings (time step, end time, saving) stay in each
//! consumer. The model is documented in `examples/lahar_multifield.rs`.
//!
//! Requires `rsc/models/lahar_physics.rs`, `rsc/models/lahar_source.rs` and
//! `rsc/models/lahar_flux.rs` as sibling modules named `lahar_physics`,
//! `lahar_source` and `lahar_flux`.
//!
//! With the `parallel` feature, the builders take an
//! `internal_threshold: Option<usize>` for the dispatch sites of the model
//! (source term, flux, bed-slope calculator): `None` keeps each site's
//! default, `Some(t)` sets all of them to `t`.

use std::sync::Arc;

#[cfg(feature = "parallel")]
use super::lahar_flux::default_flux_parallel_threshold;
use super::lahar_flux::ShallowLayerFlux;
use super::lahar_physics::{front_position, COL_B, COL_M};
#[cfg(feature = "parallel")]
use super::lahar_source::default_source_parallel_threshold;
use super::lahar_source::{build_source, validation_physics, ValidationRegime};
use oxiflow::{
    context::{
        calculators::{FDGradientCalculator, FDScheme},
        error::OxiflowError,
    },
    mesh::{Mesh, UniformGrid1D},
    model::{CompositeModel, DiscretizedModel},
    operators::limiters::Limiter,
    solver::{scenario::Scenario, SimulationResult},
};

/// Source rate per unit width [m²/s] (paper Sec. 4.1, `q = 1`).
pub(crate) const SOURCE_RATE: f64 = 1.0;

/// Grid of `n_points` nodes over `[-x_half_width, x_half_width]`, the source
/// strip centred on `x = 0`.
pub(crate) fn grid(n_points: usize, x_half_width: f64) -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(n_points, -x_half_width, x_half_width)
}

/// The model of `regime` on [`grid`]: the source term then the flux, starting
/// from a dry bed.
pub(crate) fn model(
    regime: ValidationRegime,
    n_points: usize,
    x_half_width: f64,
    #[cfg(feature = "parallel")] internal_threshold: Option<usize>,
) -> Result<CompositeModel, OxiflowError> {
    let source = build_source(
        &grid(n_points, x_half_width)?,
        regime,
        SOURCE_RATE,
        #[cfg(feature = "parallel")]
        internal_threshold.unwrap_or_else(default_source_parallel_threshold),
    );
    let flux = ShallowLayerFlux {
        physics: validation_physics(),
        limiter: Limiter::MinMod,
        #[cfg(feature = "parallel")]
        parallel_threshold: internal_threshold.unwrap_or_else(default_flux_parallel_threshold),
    };
    let discretized = DiscretizedModel::new(
        Arc::new(flux),
        Arc::new(grid(n_points, x_half_width)?),
        "lahar_flux",
    );
    CompositeModel::new(
        vec![Box::new(source), Box::new(discretized)],
        "lahar_multifield",
    )
}

/// The single-domain scenario of `regime`: [`model`] on [`grid`].
pub(crate) fn scenario(
    regime: ValidationRegime,
    n_points: usize,
    x_half_width: f64,
    #[cfg(feature = "parallel")] internal_threshold: Option<usize>,
) -> Result<Scenario, OxiflowError> {
    let model = model(
        regime,
        n_points,
        x_half_width,
        #[cfg(feature = "parallel")]
        internal_threshold,
    )?;
    Ok(Scenario::single(
        Box::new(model),
        Box::new(grid(n_points, x_half_width)?),
    ))
}

/// Gradient of the bed elevation on [`grid`], read by the source term.
pub(crate) fn bed_gradient(
    n_points: usize,
    x_half_width: f64,
    #[cfg(feature = "parallel")] internal_threshold: Option<usize>,
) -> Result<FDGradientCalculator, OxiflowError> {
    let mesh: Arc<dyn Mesh> = Arc::new(grid(n_points, x_half_width)?);
    let calculator = FDGradientCalculator::new(mesh, 0, Some(COL_B), FDScheme::Central);
    #[cfg(feature = "parallel")]
    let calculator = match internal_threshold {
        Some(threshold) => calculator.with_parallel_threshold(threshold),
        None => calculator,
    };
    Ok(calculator)
}

/// Front position `(t, x_f)` of every recorded state of `result`, on `grid`.
// The benchmark times the run alone and does not read the front.
#[allow(dead_code)]
pub(crate) fn front_trajectory(
    result: &SimulationResult,
    grid: &UniformGrid1D,
) -> Result<Vec<(f64, f64)>, OxiflowError> {
    let x_coords: Vec<f64> = (0..grid.n_dof()).map(|i| grid.coordinates(i)[0]).collect();
    let rho_f = validation_physics().rho_f;
    result
        .times
        .iter()
        .zip(&result.states)
        .map(|(&t, state)| {
            let m_col = state.as_vector_field()?.column(COL_M).into_owned();
            Ok((t, front_position(&m_col, &x_coords, rho_f)))
        })
        .collect()
}
