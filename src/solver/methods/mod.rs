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
//!
//! ## Reserved — J4 (v0.4.0)
//!
//! | Method | Type | Note |
//! |---|---|---|
//! | `IMEXSolver` | Strang splitting | Transport-reaction |
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
//! `compute_physics`. [`evaluate_derivative`] implements this contract once
//! so each solver doesn't reimplement it per stage.
//!
//! Implicit methods ([`backward_euler`], [`crank_nicolson`], [`bdf2`]) build
//! on the same [`evaluate_derivative`] for their explicit-derivative
//! evaluation, plus the shared machinery in [`implicit`] (frozen-Jacobian
//! Newton correction, DD-033) for the implicit part.
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

pub mod backward_euler;
pub mod bdf2;
pub mod crank_nicolson;
pub mod dopri45;
pub mod euler;
pub mod implicit;
pub mod rk4;
pub mod step_control;

pub use backward_euler::BackwardEulerSolver;
pub use bdf2::BDF2Solver;
pub use crank_nicolson::CrankNicolsonSolver;
pub use dopri45::DoPri45Solver;
pub use euler::ForwardEulerSolver;
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

/// Checks that all nodal values in `state` are finite.
///
/// Shared between integrators — returns `OxiflowError::SolverDivergence` on
/// the first non-finite value found.
pub(crate) fn check_finite(state: &ContextValue, t: f64) -> Result<(), OxiflowError> {
    if let Ok(field) = state.as_scalar_field() {
        if field.iter().any(|v| !v.is_finite()) {
            return Err(OxiflowError::SolverDivergence {
                time: t,
                reason: "non-finite value detected in state vector".into(),
            });
        }
    }
    Ok(())
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
    /// (see [`evaluate_derivative`]); the returned value is the state at
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

            check_finite(&u, t_next)?;

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
