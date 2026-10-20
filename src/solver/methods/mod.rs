//! # Module `solver::methods`
//!
//! Temporal integration methods.
//!
//! ## Active at J4a (v0.4.0)
//!
//! | Method | Type | Issue |
//! |---|---|---|
//! | [`euler::ForwardEulerSolver`] | Explicit, 1st order | #33, #41 |
//! | [`rk4::RK4Solver`] | Explicit, 4th order | #41 |
//! | [`backward_euler::BackwardEulerSolver`] | Implicit, 1st order | #43, DD-013, DD-033 |
//! | [`crank_nicolson::CrankNicolsonSolver`] | Semi-implicit, 2nd order | #43, DD-013, DD-033 |
//! | [`bdf2::BDF2Solver`] | Implicit multi-step, 2nd order | #44, DD-034 |
//! | [`dopri45::DoPri45Solver`] | Adaptive explicit, order 5 | #42, DD-036 |
//! | [`imex::OperatorSplittingSolver`] | Operator splitting (Strang), n ≥ 2 | #45, DD-037 |
//!
//! All integrators are decoupled from the spatial scheme via
//! `DiscreteOperator<M: Mesh>` (INV-2, J4b) — no FD/FV/FEM
//! method is called directly inside an integrator.
//!
//! ## Shared evaluation contract
//!
//! Every integrator — single-stage (Euler) or multi-stage (RK4 and beyond) —
//! follows the same per-evaluation order documented in
//! [`crate::solver`]: calculators, then boundary conditions, then
//! `compute_physics`. `evaluate_derivative` implements this contract once
//! so each solver doesn't reimplement it per stage.
//!
//! Implicit methods ([`backward_euler`], [`crank_nicolson`], [`bdf2`]) build
//! on the same `evaluate_derivative` for their explicit-derivative
//! evaluation, plus the shared machinery in [`implicit`] for the implicit
//! part. [`newton`] provides the iterated Newton alternative DD-033
//! anticipated (DD-044, v0.7.0), wired into [`backward_euler`]/
//! [`crank_nicolson`] (#111) and [`bdf2`] (#112) — [`bdf2`]'s
//! startup/bootstrap step is the one exception, unconditionally using
//! the non-configurable single-correction path (see that module's docs).
//!
//! ## Per-step primitive, with history (DD-031, DD-034)
//!
//! [`SteppableSolver`] exposes a single-step primitive on top of `Solver`'s
//! full-time-range `solve()`. [`crate::solver::orchestrator::MultiDomainOrchestrator`]
//! calls it once per domain per synchronised step, so each domain in a
//! coupled scenario can use a different integrator. Multi-step methods
//! ([`bdf2::BDF2Solver`], needing `u^{n-1}`) declare
//! [`SteppableSolver::history_depth`] — callers maintain a buffer of that
//! many past states and pass it to [`SteppableSolver::step`]. One-step
//! methods use the default (`0`) and ignore the parameter (DD-034).
//!
//! [`SteppableSolver::solve_fixed_step`] (DD-035) factors out the common
//! `solve()` body every fixed-step solver above used to duplicate
//! near-verbatim — each one's `Solver::solve()` is now a single call to
//! it. [`dopri45::DoPri45Solver`] does **not** use this: variable-step
//! methods need their own loop (`dt` changes every iteration, no fixed
//! total step count) and implement `Solver` directly instead — see
//! [`step_control`] (DD-036) for the shared piece adaptive methods *do*
//! get: the step-size controller itself, decoupled from any specific
//! integrator's error source.
//!
//! [`imex::OperatorSplittingSolver`] (DD-037, #45) also implements `Solver`
//! directly rather than `SteppableSolver` — same posture as DoPri45: each
//! of its sub-operators already carries its own `Domain`, so fitting it
//! into the single-`Domain`-per-step shape of `SteppableSolver::step`
//! would reopen the multi-domain orchestration question for a need #45
//! never asked for.

pub mod backward_euler;
pub mod bdf2;
pub mod crank_nicolson;
pub mod dopri45;
pub mod euler;
pub mod imex;
pub mod implicit;
pub mod newton;
pub mod rk4;
pub mod step_control;

pub use backward_euler::BackwardEulerSolver;
pub use bdf2::BDF2Solver;
pub use crank_nicolson::CrankNicolsonSolver;
pub use dopri45::DoPri45Solver;
pub use euler::ForwardEulerSolver;
pub use imex::{OperatorSplittingSolver, SplitOperator, SplittingScheme};
pub use newton::{JacobianStrategy, NewtonConvergence};
pub use rk4::RK4Solver;
pub use step_control::StepSizeController;

use std::collections::HashMap;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::StepControl;
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::snapshot::SimulationSnapshot;
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// Applies every boundary condition attached to `domain` to `state`, in
/// declaration order.
///
/// Called once per derivative evaluation, after context calculators and
/// before `compute_physics` — see [`crate::boundary::BoundaryCondition::apply`].
/// For multi-stage integrators this means once *per stage*, on that stage's
/// intermediate state: a Dirichlet value must be re-enforced at every
/// evaluation point, not just once per outer time step, or it would drift
/// across stages.
///
/// No-op if `domain` has no boundary conditions attached (J1-style scenarios).
pub(crate) fn apply_boundary_conditions(
    domain: &Domain,
    state: &mut ContextValue,
    ctx: &ComputeContext,
) -> Result<(), OxiflowError> {
    if domain.boundary_conditions.is_empty() {
        return Ok(());
    }
    let field = state.as_scalar_field_mut()?;
    for bc in &domain.boundary_conditions {
        bc.apply(field, ctx, domain.mesh.as_ref())?;
    }
    Ok(())
}

/// Evaluates `du/dt` for `domain` at time `t`, following the contractual
/// order: build `ComputeContext`, run the calculator chain, apply boundary
/// conditions, then call `compute_physics`.
///
/// `state` is mutated in-place by boundary condition application — this is
/// intentional (see [`apply_boundary_conditions`]) and mirrors the contract
/// documented on [`crate::boundary::BoundaryCondition::apply`]. Callers pass
/// either the persisted solution state (single-stage methods) or a
/// transient intermediate stage state (multi-stage methods); in both cases
/// the corrected state is also what `compute_physics` consumes.
pub(crate) fn evaluate_derivative(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
) -> Result<ContextValue, OxiflowError> {
    let mut ctx = ComputeContext::new(t, dt);

    for calc in chain {
        let value = calc
            .compute(state, &ctx)
            .map_err(|e| OxiflowError::ComputationFailed {
                variable: calc.provides(),
                source: Box::new(e),
            })?;
        ctx.insert(calc.provides(), value);
    }

    apply_boundary_conditions(domain, state, &ctx)?;

    domain.model.compute_physics(state, &ctx)
}

/// Checks that every value in `state` is finite.
///
/// Shared between integrators — returns `OxiflowError::SolverDivergence`
/// when a NaN or an infinity is found, whatever the numeric variant of the
/// state: a `VectorField` is checked component by component, like a
/// `ScalarField` (#192). A `Boolean` state has nothing to check.
pub(crate) fn check_finite(state: &ContextValue, t: f64) -> Result<(), OxiflowError> {
    let finite = match state {
        ContextValue::Scalar(x) => x.is_finite(),
        ContextValue::Boolean(_) => true,
        ContextValue::Vector(v) | ContextValue::ScalarField(v) => v.iter().all(|x| x.is_finite()),
        ContextValue::Matrix(m) | ContextValue::VectorField(m) => m.iter().all(|x| x.is_finite()),
    };
    if finite {
        Ok(())
    } else {
        Err(OxiflowError::SolverDivergence {
            time: t,
            reason: "non-finite value detected in state vector".into(),
        })
    }
}

/// Snapshot handed to [`Solver::on_divergence`] when a run stops on
/// divergence after `n_steps` steps: the offending `state` at time `t`, the
/// `error` that stopped the run, and the states saved so far as a partial
/// result.
///
/// Shared by every solver loop (fixed-step, DoPri45, operator splitting) so
/// that the snapshot contract stays the same everywhere (DD-025, #134).
pub(crate) fn divergence_snapshot(
    config: &SolverConfiguration,
    state: &ContextValue,
    t: f64,
    n_steps: usize,
    error: String,
    states: &[ContextValue],
    times: &[f64],
) -> SimulationSnapshot {
    SimulationSnapshot {
        time_config: config.time.clone(),
        integrator: config.integrator.clone(),
        state: state.clone(),
        t,
        n_steps,
        error: Some(error),
        partial: Some(SimulationResult {
            states: states.to_vec(),
            times: times.to_vec(),
            n_steps,
            metadata: HashMap::new(),
        }),
    }
}

// ── SteppableSolver ───────────────────────────────────────────────────────────

/// Single-domain solvers that expose a per-step primitive (DD-031, DD-034).
///
/// `Solver::solve()` drives a full time range for one domain; `step()`
/// advances a single domain by exactly one `dt`, given its current state
/// and (for multi-step methods) a bounded history of past states. This is
/// what [`crate::solver::orchestrator::MultiDomainOrchestrator`] calls
/// once per domain per synchronised step, allowing each domain in a
/// coupled scenario to use a different integrator.
///
/// Implementations should extract their existing single-step logic from
/// `solve()` rather than duplicate it — `solve()` is expected to call
/// `step()` internally.
pub trait SteppableSolver: Solver {
    /// Number of past states (beyond the current one) this solver needs.
    ///
    /// `0` for one-step methods (Euler, RK4, Backward Euler,
    /// Crank-Nicolson) — the default. `1` for BDF2 (needs `u^{n-1}`).
    /// Callers (the per-solver `solve()` loop, or
    /// [`crate::solver::orchestrator::MultiDomainOrchestrator`]) are
    /// responsible for maintaining a buffer of exactly this many past
    /// states and passing it to [`step`](Self::step).
    fn history_depth(&self) -> usize {
        0
    }

    /// Advances `state` by one step of size `dt`, at time `t`, for `domain`.
    ///
    /// `chain` is the calculator chain already built for this run (built
    /// once by the caller, reused across steps — see
    /// [`crate::solver::chain::build_calculator_chain`]).
    ///
    /// `history` holds up to [`history_depth`](Self::history_depth) past
    /// states, most recent first (`history[0]` is `u^{n-1}`) — shorter
    /// than that during startup (empty on the very first step). Solvers
    /// with `history_depth() == 0` ignore this parameter entirely.
    ///
    /// `state` may be mutated in-place by boundary condition application
    /// (see `evaluate_derivative`); the returned value is the state at
    /// `t + dt`.
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        history: &[ContextValue],
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError>;

    /// Drives a full fixed-step time range, calling [`step`](Self::step)
    /// in a loop and managing the history buffer generically via
    /// [`history_depth`](Self::history_depth) — the common body every
    /// fixed-step solver (Euler, RK4, Backward Euler, Crank-Nicolson,
    /// BDF2) used to duplicate near-verbatim.
    ///
    /// `Solver::solve()` implementations for fixed-step methods are
    /// expected to be a single call to this: `self.solve_fixed_step(scenario, config)`.
    ///
    /// Variable-step methods (DoPri45) do **not** use this — they need
    /// their own loop (`dt` changes every iteration, total step count
    /// isn't known upfront) and implement `Solver::solve()` directly
    /// instead.
    ///
    /// # Errors
    ///
    /// `OxiflowError::InvalidDomain` if `config.time.step_control` is not
    /// `StepControl::Fixed`, if `dt <= 0.0`, or if `t_end <= t_start`.
    fn solve_fixed_step(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        scenario.validate()?;
        let domain = scenario.single_domain()?;

        let dt = match &config.time.step_control {
            StepControl::Fixed { dt } => *dt,
            _ => {
                return Err(OxiflowError::InvalidDomain(
                    "this solver only supports StepControl::Fixed (adaptive step control \
                     not supported by this method)"
                        .into(),
                ))
            }
        };

        let t_end = config.time.t_end;
        let t_start = scenario.t_start;

        if dt <= 0.0 {
            return Err(OxiflowError::InvalidDomain(
                "dt must be strictly positive".into(),
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
        let mut history: Vec<ContextValue> = Vec::new();

        let n_steps = ((t_end - t_start) / dt).round() as usize;
        let save_every = config.time.save_every.unwrap_or(1);
        let capacity = n_steps / save_every + 1;
        let mut states: Vec<ContextValue> = Vec::with_capacity(capacity);
        let mut times: Vec<f64> = Vec::with_capacity(capacity);

        states.push(u.clone());
        times.push(t_start);

        let depth = self.history_depth();

        for step in 0..n_steps {
            let t = t_start + (step as f64) * dt;
            let t_next = t_start + ((step + 1) as f64) * dt;

            let next = self.step(domain, &chain, &mut u, &history, t, dt)?;

            if depth > 0 {
                // `u` was mutated in-place by BC application inside
                // `step()` above -- push *that* corrected value, not a
                // pre-correction one. `mem::replace` avoids a clone: `u`
                // is moved into history, `next` takes its place.
                let prev = std::mem::replace(&mut u, next);
                history.insert(0, prev);
                history.truncate(depth);
            } else {
                u = next;
            }

            if let Err(e) = check_finite(&u, t_next) {
                self.on_divergence(&divergence_snapshot(
                    config,
                    &u,
                    t_next,
                    step + 1,
                    e.to_string(),
                    &states,
                    &times,
                ));
                return Err(e);
            }

            if (step + 1) % save_every == 0 {
                states.push(u.clone());
                times.push(t_next);
            }
        }

        Ok(SimulationResult {
            states,
            times,
            n_steps,
            metadata: HashMap::new(),
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use super::*;

    fn diverges(state: ContextValue) -> bool {
        matches!(
            check_finite(&state, 1.5),
            Err(OxiflowError::SolverDivergence { time, .. }) if time == 1.5
        )
    }

    #[test]
    fn check_finite_accepts_finite_states_of_every_variant() {
        for state in [
            ContextValue::Scalar(1.0),
            ContextValue::Boolean(false),
            ContextValue::Vector(DVector::from_element(3, 1.0)),
            ContextValue::Matrix(DMatrix::from_element(2, 2, 1.0)),
            ContextValue::ScalarField(DVector::from_element(3, 1.0)),
            ContextValue::VectorField(DMatrix::from_element(3, 2, 1.0)),
        ] {
            assert!(check_finite(&state, 0.0).is_ok(), "{state:?}");
        }
    }

    #[test]
    fn check_finite_detects_nan_and_infinity_in_scalar_field() {
        assert!(diverges(ContextValue::ScalarField(DVector::from_vec(
            vec![0.0, f64::NAN]
        ))));
        assert!(diverges(ContextValue::ScalarField(DVector::from_vec(
            vec![f64::NEG_INFINITY, 0.0]
        ))));
    }

    #[test]
    fn check_finite_detects_non_finite_value_in_any_vector_field_component() {
        // Before #192, a VectorField state was never inspected.
        for (row, col) in [(0, 0), (2, 1)] {
            let mut m = DMatrix::from_element(3, 2, 1.0);
            m[(row, col)] = f64::INFINITY;
            assert!(diverges(ContextValue::VectorField(m)), "({row}, {col})");
        }
    }

    #[test]
    fn check_finite_detects_non_finite_scalar_vector_and_matrix() {
        assert!(diverges(ContextValue::Scalar(f64::NAN)));
        assert!(diverges(ContextValue::Vector(DVector::from_vec(vec![
            f64::INFINITY
        ]))));
        assert!(diverges(ContextValue::Matrix(DMatrix::from_element(
            1,
            1,
            f64::NAN
        ))));
    }

    #[test]
    fn divergence_snapshot_carries_state_error_and_partial_result() {
        use crate::solver::config::{IntegratorKind, StepControl, TimeConfiguration};

        let config = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }),
            IntegratorKind::Euler,
        );
        let state = ContextValue::VectorField(DMatrix::from_element(2, 2, f64::NAN));
        let saved = vec![ContextValue::VectorField(DMatrix::from_element(2, 2, 1.0))];
        let snapshot =
            divergence_snapshot(&config, &state, 0.3, 3, "diverged".into(), &saved, &[0.0]);

        assert_eq!(snapshot.t, 0.3);
        assert_eq!(snapshot.n_steps, 3);
        assert_eq!(snapshot.error.as_deref(), Some("diverged"));
        assert_eq!(snapshot.integrator, IntegratorKind::Euler);
        assert_eq!(snapshot.time_config.t_end, 1.0);
        assert!(snapshot.state.as_vector_field().unwrap()[(0, 0)].is_nan());
        let partial = snapshot.partial.unwrap();
        assert_eq!(partial.states, saved);
        assert_eq!(partial.times, vec![0.0]);
        assert_eq!(partial.n_steps, 3);
    }
}
