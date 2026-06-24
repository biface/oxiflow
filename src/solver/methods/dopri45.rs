//! # Module `solver::methods::dopri45`
//!
//! Dormand-Prince DoPri45 — explicit, adaptive step, order 5 (local
//! extrapolation) with embedded order-4 error estimate (issue #42).
//!
//! ## Algorithm
//!
//! 7-stage embedded Runge-Kutta pair. The FSAL (First Same As Last)
//! property of the Dormand-Prince tableau means stage 7's evaluation
//! point **is** the 5th-order solution — no separate weighted sum needed
//! to obtain it (see `dopri45_stages` below). The 4th-order solution is
//! never computed explicitly either: only the *difference*
//! `b_i - b*_i` is needed for the error estimate.
//!
//! Butcher tableau coefficients verified by hand before writing this file
//! (no compiler available in the authoring environment): row sums
//! `sum_j a_ij = c_i` for every stage, `sum(b) = 1`, `sum(b*) = 1`. These
//! are necessary consistency conditions for any valid RK tableau — they
//! don't guarantee every individual coefficient is correct, but they
//! would catch most transcription errors. Cross-check against a
//! published source (Hairer/Nørsett/Wanner, or Dormand & Prince 1980)
//! before trusting this in production.
//!
//! ## Step-size control (DD-036)
//!
//! Delegates entirely to [`StepSizeController`](super::step_control::StepSizeController)
//! for the accept/reject decision and the next `dt` — this file only
//! supplies the error norm input (the RK4/5 difference) and the rejection
//! retry loop. See [`super::step_control`] for the controller itself.
//!
//! ## Scope — `Solver` only, not `SteppableSolver` (see discussion)
//!
//! Unlike BDF2 (DD-034), this solver does **not** implement
//! `SteppableSolver`. BDF2's gap (needing `u^{n-1}`) was orthogonal to
//! `dt` itself; this solver's defining feature is choosing its *own*
//! `dt` across calls, which is in direct tension with
//! `MultiDomainOrchestrator`'s v1 scope (DD-031: `dt` synchronised across
//! domains). Making this orchestrator-compatible would mean re-opening
//! the multirate question DD-031 already deferred — not a gap orthogonal
//! to an existing limitation, but the same one. Revisit together if
//! multirate coupling is ever tackled.
//!
//! ## Acceptance criteria mapping (#42)
//!
//! - Step-size control within `rtol`/`atol`: enforced by
//!   `StepSizeController::accept`.
//! - `dt_min` guard -> `OxiflowError::SolverDivergence`: the rejection
//!   retry loop below bails out once the controller can no longer
//!   suggest a smaller `dt`.
//! - Accepted/rejected counts in `SimulationResult::metadata`: keys
//!   `"solver.accepted_steps"` / `"solver.rejected_steps"`, following the
//!   convention already documented on `SimulationResult`.
//! - Performance comparable to fixed-step RK4 on non-stiff problems: not
//!   benchmarked here (no compiler in this environment) — worth checking
//!   once this runs for real.

use std::collections::HashMap;

use nalgebra::DVector;

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::StepControl;
use crate::solver::methods::step_control::StepSizeController;
use crate::solver::methods::{check_finite, evaluate_derivative};
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

// ── Dormand-Prince Butcher tableau ──────────────────────────────────────────────

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
const C6: f64 = 1.0;
const C7: f64 = 1.0;

const A21: f64 = 1.0 / 5.0;

const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;

const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;

const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;

const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

const A71: f64 = 35.0 / 384.0;
// A72 = 0.0 -- the k2 term drops out of the stage-7 (and 5th-order
// solution) combination entirely.
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;

/// Error weights `b_i - b*_i` (5th-order minus 4th-order), precomputed by
/// hand (see module docs for the derivation and consistency check).
const E1: f64 = 71.0 / 57600.0;
const E2: f64 = 0.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Order of the embedded (lower-order) error estimator -- used by the
/// step-size controller's exponents (DD-036).
const ERROR_ESTIMATOR_ORDER: f64 = 4.0;

/// Safety guard against an infinite rejection loop on a single step --
/// the controller-driven `dt`-shrink should converge well before this in
/// any well-posed problem; hitting it means tolerance genuinely cannot be
/// satisfied (acceptance criterion, #42).
const MAX_REJECTIONS_PER_STEP: usize = 50;

/// Dormand-Prince DoPri45 solver — explicit, adaptive step.
///
/// See [module docs](self) for the algorithm, the step-size controller
/// delegation, and why this implements `Solver` only (not
/// `SteppableSolver`).
pub struct DoPri45Solver;

impl Solver for DoPri45Solver {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        scenario.validate()?;
        let domain = scenario.single_domain()?;

        let (dt_init, dt_min, dt_max, rtol, atol) = match &config.time.step_control {
            StepControl::Adaptive {
                dt_init,
                dt_min,
                dt_max,
                rtol,
                atol,
            } => (*dt_init, *dt_min, *dt_max, *rtol, *atol),
            _ => {
                return Err(OxiflowError::InvalidDomain(
                    "DoPri45Solver only supports StepControl::Adaptive".into(),
                ))
            }
        };

        let t_end = config.time.t_end;
        let t_start = scenario.t_start;

        if dt_init <= 0.0 || dt_min <= 0.0 || dt_max < dt_min {
            return Err(OxiflowError::InvalidDomain(
                "dt_init and dt_min must be strictly positive, and dt_max >= dt_min".into(),
            ));
        }
        if t_end <= t_start {
            return Err(OxiflowError::InvalidDomain(
                "t_end must be greater than t_start".into(),
            ));
        }

        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &config.calculators)?;

        let mut u = domain.model.initial_state(domain.mesh.as_ref());
        let mut t = t_start;
        let mut dt = dt_init;

        let mut controller =
            StepSizeController::new(rtol, atol, dt_min, dt_max, ERROR_ESTIMATOR_ORDER);

        // `save_every` counts *accepted* steps -- there's no fixed total
        // step count to divide by here, unlike `solve_fixed_step`.
        let save_every = config.time.save_every.unwrap_or(1);
        let mut accepted_since_save = 0usize;

        let mut states: Vec<ContextValue> = vec![u.clone()];
        let mut times: Vec<f64> = vec![t_start];

        let mut accepted_steps = 0usize;
        let mut rejected_steps = 0usize;

        // Small tolerance against floating-point accumulation in `t`
        // (unavoidable here, unlike the fixed-step solvers: `dt` varies,
        // so there's no `t_start + step * dt` closed form to fall back
        // on -- see euler.rs's module docs for that fix, which doesn't
        // apply to variable-step methods).
        while t < t_end - 1e-12 {
            let mut local_dt = dt.min(t_end - t);
            let mut rejections_this_step = 0usize;

            loop {
                let mut attempt_state = u.clone();
                let (y5_field, error_field) =
                    dopri45_stages(domain, &chain, &mut attempt_state, t, local_dt)?;

                let error_norm = controller.error_norm(&error_field, &y5_field);

                if controller.accept(error_norm) {
                    u = ContextValue::ScalarField(y5_field);
                    t += local_dt;
                    accepted_steps += 1;
                    accepted_since_save += 1;

                    check_finite(&u, t)?;

                    dt = controller.next_dt(local_dt, error_norm);

                    if accepted_since_save >= save_every {
                        states.push(u.clone());
                        times.push(t);
                        accepted_since_save = 0;
                    }
                    break;
                }

                rejected_steps += 1;
                rejections_this_step += 1;
                let suggested = controller.next_dt(local_dt, error_norm);

                if rejections_this_step >= MAX_REJECTIONS_PER_STEP
                    || suggested <= controller.dt_min()
                {
                    return Err(OxiflowError::SolverDivergence {
                        time: t,
                        reason: format!(
                            "step rejected {rejections_this_step} times in a row; cannot \
                             satisfy rtol/atol even at dt_min={}",
                            controller.dt_min()
                        ),
                    });
                }

                local_dt = suggested;
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("solver.accepted_steps".to_string(), accepted_steps as f64);
        metadata.insert("solver.rejected_steps".to_string(), rejected_steps as f64);

        Ok(SimulationResult {
            states,
            times,
            n_steps: accepted_steps,
            metadata,
        })
    }
}

/// Computes the 7 Dormand-Prince stages and returns `(y_5th_order,
/// error_estimate)`.
///
/// `state` (`u^n`) is mutated in-place by boundary condition application
/// at stage 1, same contract as every other solver in this crate (see
/// [`evaluate_derivative`]).
fn dopri45_stages(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
) -> Result<(DVector<f64>, DVector<f64>), OxiflowError> {
    // Stage 1.
    let k1_val = evaluate_derivative(domain, chain, state, t, dt)?;
    let u_field = state.as_scalar_field()?.clone();
    let k1 = k1_val.as_scalar_field()?.clone();

    // Stage 2.
    let y2 = u_field.clone() + k1.clone() * (dt * A21);
    let mut s2 = ContextValue::ScalarField(y2);
    let k2_val = evaluate_derivative(domain, chain, &mut s2, t + C2 * dt, dt)?;
    let k2 = k2_val.as_scalar_field()?.clone();

    // Stage 3.
    let y3 = u_field.clone() + k1.clone() * (dt * A31) + k2.clone() * (dt * A32);
    let mut s3 = ContextValue::ScalarField(y3);
    let k3_val = evaluate_derivative(domain, chain, &mut s3, t + C3 * dt, dt)?;
    let k3 = k3_val.as_scalar_field()?.clone();

    // Stage 4.
    let y4 = u_field.clone()
        + k1.clone() * (dt * A41)
        + k2.clone() * (dt * A42)
        + k3.clone() * (dt * A43);
    let mut s4 = ContextValue::ScalarField(y4);
    let k4_val = evaluate_derivative(domain, chain, &mut s4, t + C4 * dt, dt)?;
    let k4 = k4_val.as_scalar_field()?.clone();

    // Stage 5.
    let y5s = u_field.clone()
        + k1.clone() * (dt * A51)
        + k2.clone() * (dt * A52)
        + k3.clone() * (dt * A53)
        + k4.clone() * (dt * A54);
    let mut s5 = ContextValue::ScalarField(y5s);
    let k5_val = evaluate_derivative(domain, chain, &mut s5, t + C5 * dt, dt)?;
    let k5 = k5_val.as_scalar_field()?.clone();

    // Stage 6.
    let y6 = u_field.clone()
        + k1.clone() * (dt * A61)
        + k2.clone() * (dt * A62)
        + k3.clone() * (dt * A63)
        + k4.clone() * (dt * A64)
        + k5.clone() * (dt * A65);
    let mut s6 = ContextValue::ScalarField(y6);
    let k6_val = evaluate_derivative(domain, chain, &mut s6, t + C6 * dt, dt)?;
    let k6 = k6_val.as_scalar_field()?.clone();

    // Stage 7 -- A72 is 0.0, the k2 term is omitted entirely. The state
    // this stage is evaluated at, `y7`, *is* the 5th-order solution
    // (FSAL: b_i == a7i for i=1..6, b7 == 0 -- see module docs).
    let y7 = u_field.clone()
        + k1.clone() * (dt * A71)
        + k3.clone() * (dt * A73)
        + k4.clone() * (dt * A74)
        + k5.clone() * (dt * A75)
        + k6.clone() * (dt * A76);
    let mut s7 = ContextValue::ScalarField(y7.clone());
    let k7_val = evaluate_derivative(domain, chain, &mut s7, t + C7 * dt, dt)?;
    let k7 = k7_val.as_scalar_field()?.clone();

    let y_5th = y7;

    // Error estimate: dt * sum((b_i - b*_i) * k_i).
    let error: DVector<f64> = k1 * (dt * E1)
        + k2 * (dt * E2)
        + k3 * (dt * E3)
        + k4 * (dt * E4)
        + k5 * (dt * E5)
        + k6 * (dt * E6)
        + k7 * (dt * E7);

    Ok((y_5th, error))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::config::{IntegratorKind, SolverConfiguration, TimeConfiguration};
    use crate::solver::scenario::Scenario;

    #[derive(Debug)]
    struct ExponentialDecay {
        lambda: f64,
    }

    impl RequiresContext for ExponentialDecay {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for ExponentialDecay {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(u.map(|v| -self.lambda * v)))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }

        fn name(&self) -> &str {
            "exponential_decay"
        }
    }

    #[derive(Debug)]
    struct ZeroDerivative;

    impl RequiresContext for ZeroDerivative {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for ZeroDerivative {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(DVector::zeros(u.len())))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 2.5))
        }

        fn name(&self) -> &str {
            "zero_derivative"
        }
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    fn make_config(t_end: f64, dt_init: f64, rtol: f64, atol: f64) -> SolverConfiguration {
        SolverConfiguration::new(
            TimeConfiguration::new(
                t_end,
                StepControl::Adaptive {
                    dt_init,
                    dt_min: 1e-8,
                    dt_max: 1.0,
                    rtol,
                    atol,
                },
            ),
            IntegratorKind::DoPri45,
        )
    }

    // ── Butcher tableau consistency (catches transcription errors) ───────────

    #[test]
    fn row_sums_match_c_nodes() {
        // sum_j a_ij == c_i for every stage -- a necessary (not
        // sufficient) condition for a consistent RK tableau.
        assert!((A21 - C2).abs() < 1e-12);
        assert!(((A31 + A32) - C3).abs() < 1e-12);
        assert!(((A41 + A42 + A43) - C4).abs() < 1e-12);
        assert!(((A51 + A52 + A53 + A54) - C5).abs() < 1e-12);
        assert!(((A61 + A62 + A63 + A64 + A65) - C6).abs() < 1e-12);
        assert!(((A71 + A73 + A74 + A75 + A76) - C7).abs() < 1e-12); // A72 = 0
    }

    #[test]
    fn fifth_order_weights_sum_to_one() {
        // b7 = 0 (FSAL), b2 = 0 -- only b1,b3,b4,b5,b6 are nonzero.
        let sum = A71 + A73 + A74 + A75 + A76; // == b1+b3+b4+b5+b6 by FSAL
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn error_weights_are_nonzero_where_expected() {
        // E2 is the only zero -- b2 = b2* = 0 exactly, the rest involve
        // genuinely different 4th/5th order weights.
        assert_eq!(E2, 0.0);
        assert_ne!(E1, 0.0);
        assert_ne!(E3, 0.0);
        assert_ne!(E4, 0.0);
        assert_ne!(E5, 0.0);
        assert_ne!(E6, 0.0);
        assert_ne!(E7, 0.0);
    }

    // ── Basic correctness ─────────────────────────────────────────────────────

    #[test]
    fn zero_derivative_field_stays_constant() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(5));
        let config = make_config(1.0, 0.1, 1e-6, 1e-9);
        let result = DoPri45Solver.solve(&scenario, &config).unwrap();
        for state in &result.states {
            let field = state.as_scalar_field().unwrap();
            for v in field.iter() {
                assert!((v - 2.5).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn exponential_decay_within_tolerance() {
        // Acceptance criterion (#42): error stays within rtol/atol for a
        // reference problem.
        let lambda = 2.0;
        let rtol = 1e-8;
        let atol = 1e-10;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let config = make_config(1.0, 0.1, rtol, atol);
        let result = DoPri45Solver.solve(&scenario, &config).unwrap();

        let expected = (-lambda * 1.0_f64).exp();
        let final_field = result.states.last().unwrap().as_scalar_field().unwrap();
        for v in final_field.iter() {
            // Loose-ish bound: rtol/atol govern the *local* per-step error,
            // not a hard global guarantee -- a generous multiple of rtol
            // is the honest comparison here, not rtol itself.
            assert!(
                (v - expected).abs() < 1e-5,
                "got {v}, expected {expected} (lambda={lambda})"
            );
        }
    }

    #[test]
    fn t_final_reaches_t_end() {
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 1.0 }), make_mesh(2));
        let config = make_config(2.0, 0.1, 1e-6, 1e-9);
        let result = DoPri45Solver.solve(&scenario, &config).unwrap();
        assert!((result.t_final().unwrap() - 2.0).abs() < 1e-6);
    }

    // ── Metadata (acceptance criterion, #42) ──────────────────────────────────

    #[test]
    fn metadata_reports_accepted_and_rejected_steps() {
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 5.0 }), make_mesh(2));
        let config = make_config(1.0, 0.1, 1e-6, 1e-9);
        let result = DoPri45Solver.solve(&scenario, &config).unwrap();

        assert!(result.metadata.contains_key("solver.accepted_steps"));
        assert!(result.metadata.contains_key("solver.rejected_steps"));
        assert!(result.metadata["solver.accepted_steps"] > 0.0);
        assert_eq!(
            result.metadata["solver.accepted_steps"],
            result.n_steps as f64
        );
    }

    // ── dt_min guard (acceptance criterion, #42) ──────────────────────────────

    #[test]
    fn dt_min_guard_raises_solver_divergence() {
        // dt_init == dt_min == dt_max: the controller has zero room to
        // shrink, so any rejection is immediately unrecoverable. Extremely
        // tight tolerance forces that first rejection.
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 50.0 }), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(
                1.0,
                StepControl::Adaptive {
                    dt_init: 0.5,
                    dt_min: 0.5,
                    dt_max: 0.5,
                    rtol: 1e-15,
                    atol: 1e-15,
                },
            ),
            IntegratorKind::DoPri45,
        );

        let err = DoPri45Solver.solve(&scenario, &config).unwrap_err();
        assert!(matches!(err, OxiflowError::SolverDivergence { .. }));
    }

    // ── Validation errors ─────────────────────────────────────────────────────

    #[test]
    fn fixed_step_control_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }),
            IntegratorKind::DoPri45,
        );
        assert!(DoPri45Solver.solve(&scenario, &config).is_err());
    }

    #[test]
    fn invalid_dt_bounds_return_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(
                1.0,
                StepControl::Adaptive {
                    dt_init: 0.1,
                    dt_min: 0.5, // dt_max < dt_min -- invalid
                    dt_max: 0.2,
                    rtol: 1e-6,
                    atol: 1e-9,
                },
            ),
            IntegratorKind::DoPri45,
        );
        assert!(DoPri45Solver.solve(&scenario, &config).is_err());
    }

    #[test]
    fn t_end_before_t_start_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2)).with_t_start(5.0);
        let config = make_config(1.0, 0.1, 1e-6, 1e-9);
        assert!(DoPri45Solver.solve(&scenario, &config).is_err());
    }

    #[test]
    fn missing_calculator_returns_error() {
        #[derive(Debug)]
        struct NeedsExternal;
        impl RequiresContext for NeedsExternal {
            fn required_variables(&self) -> Vec<ContextVariable> {
                vec![ContextVariable::External {
                    name: "missing".into(),
                }]
            }
        }
        impl PhysicalModel for NeedsExternal {
            fn compute_physics(
                &self,
                s: &ContextValue,
                _: &ComputeContext,
            ) -> Result<ContextValue, OxiflowError> {
                Ok(s.clone())
            }
            fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
                ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
            }
            fn name(&self) -> &str {
                "needs_external"
            }
        }

        let scenario = Scenario::single(Box::new(NeedsExternal), make_mesh(2));
        let config = make_config(1.0, 0.1, 1e-6, 1e-9);
        let err = DoPri45Solver.solve(&scenario, &config).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }
}
