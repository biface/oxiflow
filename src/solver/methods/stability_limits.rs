//! # Module `solver::methods::stability_limits`
//!
//! Step limits that stability puts on the explicit integrators
//! ([DA-073](https://github.com/biface/oxiflow/issues/229)).
//!
//! ## Advective limit
//!
//! An explicit integrator advancing an advective term `∇·F(u)` is stable
//! only if information does not travel more than one cell per step:
//!
//! $$C = \frac{s \cdot \Delta t}{\Delta x} \leq 1$$
//!
//! where `s` is the largest absolute wave speed the model declares through
//! [`PhysicalModel::max_wave_speed`](crate::model::traits::PhysicalModel::max_wave_speed),
//! `Δx` the characteristic length of the mesh
//! ([`Mesh::characteristic_length`](crate::mesh::Mesh::characteristic_length))
//! and `C` the Courant number. The limit is therefore
//! `dt_max = Δx / s`. It is a *necessary* condition, not a sufficient one:
//! forward Euler with a centred flux is unstable at any step in pure
//! advection, and RK4 with an upwind flux would tolerate a Courant number of
//! about 1.4. The limit of 1 holds for every explicit integrator of the
//! crate; a configurable limit, or one per scheme–integrator pair, would
//! come with a model that needs it.
//!
//! The limit belongs to the integrator, not to the spatial scheme: the
//! scheme sees neither the integrator nor its step control. Forward Euler
//! and RK4, with a fixed step, refuse a step above the limit
//! ([`require_within_advective_limit`]); DoPri45 shortens its step to it.
//! The implicit integrators (Backward Euler, Crank–Nicolson, BDF2) have no
//! advective limit and do not check.
//!
//! The speed is evaluated on the state at the start of each step, in the
//! context of the first stage, so the calculator chain runs once
//! (`evaluate_derivative_and_step_limit` in [`super`]).
//!
//! ## Out of scope
//!
//! The diffusive (Fourier) condition `D·Δt/Δx² ≤ 1/2` is not checked. Per-axis
//! speeds and lengths come with the nD structured grid
//! ([#152](https://github.com/biface/oxiflow/issues/152)).

use crate::context::error::OxiflowError;
use crate::solver::scenario::Domain;

/// Advective step limit `dt_max = Δx / s` of `domain` for the wave speed
/// `speed` its model declared, or `None` if the model declared no speed or
/// a zero speed (nothing is transported, no limit).
///
/// # Errors
///
/// `OxiflowError::PreconditionFailed` (context
/// `"PhysicalModel::max_wave_speed"`) if `speed` is NaN, infinite or
/// negative: such a speed cannot bound a step, and passing it on would let
/// a non-finite state through unchecked. The message names the model.
pub(crate) fn advective_step_limit(
    domain: &Domain,
    speed: Option<f64>,
) -> Result<Option<f64>, OxiflowError> {
    let Some(speed) = speed else {
        return Ok(None);
    };
    if !speed.is_finite() || speed < 0.0 {
        return Err(OxiflowError::PreconditionFailed {
            context: "PhysicalModel::max_wave_speed",
            message: format!(
                "model '{}' declared the wave speed {speed}; a wave speed must be \
                 finite and non-negative",
                domain.model.name()
            ),
        });
    }
    if speed == 0.0 {
        return Ok(None);
    }
    Ok(Some(domain.mesh.characteristic_length() / speed))
}

/// Refuses a fixed step `dt` above the advective step limit `limit`, for
/// the explicit integrator named `context`. No limit, no check.
///
/// # Errors
///
/// `OxiflowError::PreconditionFailed` with the given `context` if
/// `dt > limit`; the message gives the Courant number `dt / limit`, the
/// limit and the step.
pub(crate) fn require_within_advective_limit(
    context: &'static str,
    dt: f64,
    limit: Option<f64>,
) -> Result<(), OxiflowError> {
    match limit {
        Some(dt_max) if dt > dt_max => Err(OxiflowError::PreconditionFailed {
            context,
            message: format!(
                "advective stability limit exceeded: Courant number {:.6} > 1 \
                 (dt = {dt}, dt_max = {dt_max}); an explicit integrator needs \
                 dt <= dt_max, an implicit one has no such limit",
                dt / dt_max
            ),
        }),
        _ => Ok(()),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::UniformGrid1D;
    use crate::model::DeclaredWaveSpeed;

    /// 11 nodes on (0, 1): `Δx = 0.1`.
    fn domain(speed: f64) -> Domain {
        Domain::new(
            "d",
            Box::new(DeclaredWaveSpeed { speed }),
            Box::new(UniformGrid1D::new(11, 0.0, 1.0).unwrap()),
        )
    }

    #[test]
    fn limit_is_the_characteristic_length_over_the_speed() {
        let limit = advective_step_limit(&domain(4.0), Some(4.0))
            .unwrap()
            .unwrap();
        assert!((limit - 0.025).abs() < 1e-15);
    }

    #[test]
    fn no_speed_or_zero_speed_gives_no_limit() {
        assert_eq!(advective_step_limit(&domain(0.0), None).unwrap(), None);
        assert_eq!(advective_step_limit(&domain(0.0), Some(0.0)).unwrap(), None);
    }

    #[test]
    fn invalid_speeds_are_refused_with_the_model_name() {
        for speed in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let err = advective_step_limit(&domain(speed), Some(speed)).unwrap_err();
            match err {
                OxiflowError::PreconditionFailed { context, message } => {
                    assert_eq!(context, "PhysicalModel::max_wave_speed");
                    assert!(message.contains("declared_wave_speed"), "{message}");
                }
                other => panic!("unexpected error {other:?}"),
            }
        }
    }

    #[test]
    fn a_step_above_the_limit_is_refused_with_its_courant_number() {
        let err = require_within_advective_limit("Solver::step", 0.2, Some(0.1)).unwrap_err();
        match err {
            OxiflowError::PreconditionFailed { context, message } => {
                assert_eq!(context, "Solver::step");
                assert!(message.contains("Courant number 2.000000"), "{message}");
                assert!(message.contains("dt_max = 0.1"), "{message}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn a_step_at_or_below_the_limit_or_without_limit_passes() {
        assert!(require_within_advective_limit("Solver::step", 0.1, Some(0.1)).is_ok());
        assert!(require_within_advective_limit("Solver::step", 0.05, Some(0.1)).is_ok());
        assert!(require_within_advective_limit("Solver::step", 1e9, None).is_ok());
    }
}
