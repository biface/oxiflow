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
//!
//! ## Reserved — J4 (v0.4.0)
//!
//! | Method | Type | Note |
//! |---|---|---|
//! | `DoPri45Solver` | Adaptive explicit | `StepControl::Adaptive` |
//! | `BDF2Solver` | Implicit multi-step, 2nd order | — |
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
//! Implicit methods ([`backward_euler`], [`crank_nicolson`]) build on the
//! same [`evaluate_derivative`] for their explicit-derivative evaluation,
//! plus the shared machinery in [`implicit`] (frozen-Jacobian Newton
//! correction, DD-033) for the implicit part.
//!
//! ## Per-step primitive (DD-031)
//!
//! [`SteppableSolver`] exposes a single-step primitive on top of `Solver`'s
//! full-time-range `solve()`. [`crate::solver::orchestrator::MultiDomainOrchestrator`]
//! calls it once per domain per synchronised step, so each domain in a
//! coupled scenario can use a different integrator.

pub mod backward_euler;
pub mod crank_nicolson;
pub mod euler;
pub mod implicit;
pub mod rk4;

pub use backward_euler::BackwardEulerSolver;
pub use crank_nicolson::CrankNicolsonSolver;
pub use euler::ForwardEulerSolver;
pub use rk4::RK4Solver;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::scenario::Domain;
use crate::solver::Solver;

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

/// Single-domain solvers that expose a per-step primitive (DD-031).
///
/// `Solver::solve()` drives a full time range for one domain; `step()`
/// advances a single domain by exactly one `dt`, given its current state.
/// This is what [`crate::solver::orchestrator::MultiDomainOrchestrator`]
/// calls once per domain per synchronised step, allowing each domain in a
/// coupled scenario to use a different integrator.
///
/// Implementations should extract their existing single-step logic from
/// `solve()` rather than duplicate it — `solve()` is expected to call
/// `step()` internally.
pub trait SteppableSolver: Solver {
    /// Advances `state` by one step of size `dt`, at time `t`, for `domain`.
    ///
    /// `chain` is the calculator chain already built for this run (built
    /// once by the caller, reused across steps — see
    /// [`crate::solver::chain::build_calculator_chain`]).
    ///
    /// `state` may be mutated in-place by boundary condition application
    /// (see [`evaluate_derivative`]); the returned value is the state at
    /// `t + dt`.
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError>;
}
