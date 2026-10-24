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
//! [`crate::solver`]: the calculators the boundary conditions need, the
//! boundary conditions, the other calculators, `compute_physics`, then a
//! zero derivative at the nodes the conditions declare (DD-070, #200).
//! `evaluate_derivative` implements this contract once so each solver
//! doesn't reimplement it per stage.
//!
//! The solver drivers ([`SteppableSolver::solve_fixed_step`], the loops of
//! [`dopri45::DoPri45Solver`] and [`imex::OperatorSplittingSolver`], and
//! [`crate::solver::orchestrator::MultiDomainOrchestrator`]) also apply the
//! boundary conditions to the initial state and to the state at the end of
//! each step, before checking and saving it: every saved state satisfies
//! them. [`SteppableSolver::step`] itself does not.
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
mod state_layout;
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
use crate::context::variable::ContextVariable;
use crate::context::ContextCalculator;
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::StepControl;
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::snapshot::SimulationSnapshot;
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// Applies every boundary condition attached to `domain` to `state`, in
/// declaration order, with `ctx` holding the variables they require.
///
/// No-op if `domain` has no boundary conditions attached (J1-style scenarios).
/// Otherwise `state` must be a `ScalarField` — see the *State variants*
/// section of [`crate::boundary::BoundaryCondition`].
fn apply_boundary_conditions(
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

/// Runs one calculator of the chain on `state` and stores its value in
/// `ctx`, through [`ContextCalculator::compute_shared`]: a calculator that
/// keeps its value hands over the same allocation at every evaluation,
/// without a copy (DT-068, #195).
fn run_calculator(
    calc: &dyn ContextCalculator,
    state: &ContextValue,
    ctx: &mut ComputeContext,
) -> Result<(), OxiflowError> {
    let value = calc
        .compute_shared(state, ctx)
        .map_err(|e| OxiflowError::ComputationFailed {
            variable: calc.provides(),
            source: Box::new(e),
        })?;
    ctx.insert_shared(calc.provides(), value);
    Ok(())
}

/// Marks the calculators of `chain` that run before the boundary conditions
/// of `domain` (DD-070, #200): those providing a variable a condition
/// requires, and, through `depends_on()`, the calculators these depend on.
///
/// `chain` is in execution order, so a calculator's dependencies come
/// before it; scanning it backwards collects them transitively in one pass.
/// The relative order of the marked calculators, and of the others, is the
/// order of `chain`.
fn boundary_calculators(domain: &Domain, chain: &[&dyn ContextCalculator]) -> Vec<bool> {
    let mut needed: Vec<ContextVariable> = domain
        .boundary_conditions
        .iter()
        .flat_map(|bc| bc.required_variables())
        .collect();
    let mut marked = vec![false; chain.len()];
    for (i, calc) in chain.iter().enumerate().rev() {
        if needed.contains(&calc.provides()) {
            marked[i] = true;
            needed.extend(calc.depends_on());
        }
    }
    marked
}

/// Runs the calculators marked by [`boundary_calculators`], then applies
/// the boundary conditions of `domain` to `state`, at time `t`.
///
/// The solver drivers call it on the initial state and at the end of each
/// step (DD-070, #200), so that every saved state satisfies the conditions,
/// including a value that depends on time or on neighbouring nodes. No-op
/// if `domain` has no boundary conditions attached.
pub(crate) fn enforce_boundary_conditions(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
) -> Result<(), OxiflowError> {
    if domain.boundary_conditions.is_empty() {
        return Ok(());
    }
    let mut ctx = ComputeContext::new(t, dt);
    let marked = boundary_calculators(domain, chain);
    for (calc, _) in chain.iter().zip(&marked).filter(|(_, &m)| m) {
        run_calculator(*calc, state, &mut ctx)?;
    }
    apply_boundary_conditions(domain, state, &ctx)
}

/// Sets `derivative` to zero at the nodes the boundary conditions of
/// `domain` declare (DD-070, #200), so that no integrator or Newton
/// iteration moves them.
///
/// # Errors
///
/// `OxiflowError::PreconditionFailed` if a declared node is outside the
/// derivative; `OxiflowError::TypeMismatch` if a condition declares nodes
/// and the derivative is not a `ScalarField`.
fn hold_constrained_nodes(
    domain: &Domain,
    derivative: &mut ContextValue,
) -> Result<(), OxiflowError> {
    let mesh = domain.mesh.as_ref();
    for bc in &domain.boundary_conditions {
        let nodes = bc.constrained_nodes(mesh);
        if nodes.is_empty() {
            continue;
        }
        let field = derivative.as_scalar_field_mut()?;
        let len = field.len();
        for i in nodes {
            if i >= len {
                return Err(OxiflowError::PreconditionFailed {
                    context: "BoundaryCondition::constrained_nodes",
                    message: format!(
                        "node {i} is declared by a {:?} boundary condition, but the state \
                         has {len} nodes",
                        bc.boundary_type()
                    ),
                });
            }
            field[i] = 0.0;
        }
    }
    Ok(())
}

/// Evaluates `du/dt` for `domain` at time `t`, following the contractual
/// order (DD-070, #200):
///
/// 1. the calculators providing the variables the boundary conditions
///    require (with their dependencies);
/// 2. the boundary conditions, applied to `state`;
/// 3. the other calculators, which therefore see the constrained state;
/// 4. `compute_physics`;
/// 5. the derivative set to zero at the nodes the conditions declare.
///
/// Without boundary conditions, this is the calculator chain followed by
/// `compute_physics`.
///
/// Each calculator runs through [`ContextCalculator::compute_shared`]: a
/// calculator that keeps its value hands over the same allocation at every
/// evaluation, without a copy (DT-068, #195).
///
/// `state` is mutated in-place by boundary condition application — this is
/// intentional and mirrors the contract documented on
/// [`crate::boundary::BoundaryCondition::apply`]. Callers pass either the
/// persisted solution state (single-stage methods) or a transient
/// intermediate stage state (multi-stage methods); in both cases the
/// corrected state is also what `compute_physics` consumes. For multi-stage
/// integrators the conditions are thus re-enforced at every stage, on that
/// stage's intermediate state.
pub(crate) fn evaluate_derivative(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
) -> Result<ContextValue, OxiflowError> {
    let mut ctx = ComputeContext::new(t, dt);

    if domain.boundary_conditions.is_empty() {
        for calc in chain {
            run_calculator(*calc, state, &mut ctx)?;
        }
        return domain.model.compute_physics(state, &ctx);
    }

    let marked = boundary_calculators(domain, chain);
    for (calc, _) in chain.iter().zip(&marked).filter(|(_, &m)| m) {
        run_calculator(*calc, state, &mut ctx)?;
    }
    apply_boundary_conditions(domain, state, &ctx)?;
    for (calc, _) in chain.iter().zip(&marked).filter(|(_, &m)| !m) {
        run_calculator(*calc, state, &mut ctx)?;
    }

    let mut derivative = domain.model.compute_physics(state, &ctx)?;
    hold_constrained_nodes(domain, &mut derivative)?;
    Ok(derivative)
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
    /// `t + dt`. The boundary conditions are not re-applied to it: the
    /// drivers calling `step` do it (DD-070, #200), so a caller using `step`
    /// directly applies them itself if it needs a state that satisfies a
    /// condition whose nodes are not declared, or whose value depends on
    /// time.
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
    /// The boundary conditions are applied to the initial state and to the
    /// state returned by each step, before it is checked and saved (DD-070,
    /// #200).
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
        enforce_boundary_conditions(domain, &chain, &mut u, t_start, dt)?;
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

            let mut next = self.step(domain, &chain, &mut u, &history, t, dt)?;
            enforce_boundary_conditions(domain, &chain, &mut next, t_next, dt)?;

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

    // ── evaluate_derivative ───────────────────────────────────────────────────

    #[test]
    fn evaluate_derivative_hands_a_kept_value_to_the_model_without_copy() {
        use std::sync::{Arc, Mutex};

        use crate::context::calculator::CachedValueCalculator;
        use crate::context::compute::ComputeContext;
        use crate::context::variable::ContextVariable;
        use crate::mesh::{Mesh, UniformGrid1D};
        use crate::model::{PhysicalModel, RequiresContext};

        /// Records the address of the coefficient it reads at each evaluation.
        struct AddressRecorder {
            var: ContextVariable,
            seen: Arc<Mutex<Vec<usize>>>,
        }

        impl RequiresContext for AddressRecorder {
            fn required_variables(&self) -> Vec<ContextVariable> {
                vec![self.var.clone()]
            }
        }
        impl PhysicalModel for AddressRecorder {
            fn compute_physics(
                &self,
                state: &ContextValue,
                ctx: &ComputeContext,
            ) -> Result<ContextValue, OxiflowError> {
                let coefficient = ctx.external(self.var.clone())?;
                self.seen
                    .lock()
                    .unwrap()
                    .push(coefficient as *const ContextValue as usize);
                Ok(state.clone())
            }
            fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
                ContextValue::ScalarField(DVector::zeros(mesh.n_dof()))
            }
            fn name(&self) -> &str {
                "address recorder"
            }
        }

        let var = ContextVariable::External { name: "c".into() };
        let calc = CachedValueCalculator {
            var: var.clone(),
            value: Arc::new(ContextValue::ScalarField(DVector::from_element(5, 0.5))),
        };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let domain = Domain::new(
            "d",
            Box::new(AddressRecorder {
                var,
                seen: Arc::clone(&seen),
            }),
            Box::new(UniformGrid1D::new(5, 0.0, 1.0).unwrap()),
        );
        let chain: [&dyn ContextCalculator; 1] = [&calc];
        let mut state = ContextValue::ScalarField(DVector::zeros(5));

        evaluate_derivative(&domain, &chain, &mut state, 0.0, 0.1).unwrap();
        evaluate_derivative(&domain, &chain, &mut state, 0.1, 0.1).unwrap();

        let kept = Arc::as_ptr(&calc.value) as usize;
        assert_eq!(*seen.lock().unwrap(), vec![kept, kept]);
    }
}

// ── Boundary conditions (DD-070, #200) ───────────────────────────────────────

#[cfg(test)]
mod boundary_condition_tests {
    use std::f64::consts::PI;
    use std::sync::Arc;

    use nalgebra::DVector;

    use super::*;
    use crate::boundary::{BoundaryCondition, BoundaryType, PinnedNodes};
    use crate::context::calculators::FDLaplacianCalculator;
    use crate::context::quantity::PhysicalQuantity;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::{PhysicalModel, RequiresContext};
    use crate::solver::config::{IntegratorKind, TimeConfiguration};
    use crate::solver::orchestrator::{MultiDomainOrchestrator, OrchestratorConfig};
    use crate::solver::scenario::DomainId;

    const ALPHA: f64 = 0.1;

    fn laplacian() -> ContextVariable {
        ContextVariable::External {
            name: "laplacian".into(),
        }
    }

    /// `du/dt = alpha * d2u/dx2` on [0, 1], the Laplacian coming from
    /// `FDLaplacianCalculator`, from `u(x, 0) = 1 + sin(pi x)`. With `u = 1`
    /// held at both ends, `u = 1 + exp(-pi^2 alpha t) sin(pi x)`.
    struct Heat {
        alpha: f64,
    }

    impl RequiresContext for Heat {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![laplacian()]
        }
    }

    impl PhysicalModel for Heat {
        fn compute_physics(
            &self,
            _state: &ContextValue,
            ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let lap = ctx.external(laplacian())?.as_scalar_field()?;
            Ok(ContextValue::ScalarField(lap.map(|v| self.alpha * v)))
        }
        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_iterator(
                mesh.n_dof(),
                (0..mesh.n_dof()).map(|i| 1.0 + (PI * mesh.coordinates(i)[0]).sin()),
            ))
        }
        fn name(&self) -> &str {
            "heat"
        }
    }

    fn exact(x: f64, t: f64) -> f64 {
        1.0 + (-PI * PI * ALPHA * t).exp() * (PI * x).sin()
    }

    fn grid(n: usize) -> UniformGrid1D {
        UniformGrid1D::new(n, 0.0, 1.0).unwrap()
    }

    fn heat_domain(n: usize, alpha: f64, bc: PinnedNodes) -> Domain {
        Domain::new("bar", Box::new(Heat { alpha }), Box::new(grid(n)))
            .with_boundary_conditions(vec![Box::new(bc)])
    }

    fn laplacian_calculator(n: usize) -> Box<dyn ContextCalculator> {
        Box::new(FDLaplacianCalculator::new(Arc::new(grid(n)), laplacian()))
    }

    fn config(
        n: usize,
        kind: IntegratorKind,
        step: StepControl,
        t_end: f64,
    ) -> SolverConfiguration {
        SolverConfiguration::new(TimeConfiguration::new(t_end, step), kind)
            .with_calculator(laplacian_calculator(n))
    }

    /// Stable explicit step: a fifth of the diffusion limit `dx^2 / alpha`.
    fn explicit_dt(n: usize) -> f64 {
        let dx = 1.0 / (n - 1) as f64;
        0.2 * dx * dx / ALPHA
    }

    /// The seven solvers, with a step suited to each.
    fn solvers(
        n: usize,
        bc: &dyn Fn() -> PinnedNodes,
    ) -> Vec<(&'static str, Box<dyn Solver>, SolverConfiguration)> {
        use crate::solver::methods::{
            BDF2Solver, BackwardEulerSolver, CrankNicolsonSolver, DoPri45Solver,
            ForwardEulerSolver, OperatorSplittingSolver, RK4Solver,
        };
        let t_end = 0.2;
        let fixed = |dt| StepControl::Fixed { dt };
        let explicit = explicit_dt(n);
        let splitting = OperatorSplittingSolver::strang(
            heat_domain(n, ALPHA / 2.0, bc()),
            Box::new(ForwardEulerSolver),
            heat_domain(n, ALPHA / 2.0, bc()),
            Box::new(ForwardEulerSolver),
        )
        .unwrap();
        vec![
            (
                "Euler",
                Box::new(ForwardEulerSolver),
                config(n, IntegratorKind::Euler, fixed(explicit), t_end),
            ),
            (
                "RK4",
                Box::new(RK4Solver),
                config(n, IntegratorKind::RK4, fixed(explicit), t_end),
            ),
            (
                "BackwardEuler",
                Box::new(BackwardEulerSolver::new()),
                config(n, IntegratorKind::BackwardEuler, fixed(1e-2), t_end),
            ),
            (
                "CrankNicolson",
                Box::new(CrankNicolsonSolver::new()),
                config(n, IntegratorKind::CrankNicolson, fixed(1e-2), t_end),
            ),
            (
                "BDF2",
                Box::new(BDF2Solver::new()),
                config(n, IntegratorKind::BDF2, fixed(1e-2), t_end),
            ),
            (
                "DoPri45",
                Box::new(DoPri45Solver),
                config(
                    n,
                    IntegratorKind::DoPri45,
                    StepControl::Adaptive {
                        dt_init: explicit,
                        dt_min: 1e-9,
                        dt_max: explicit,
                        rtol: 1e-6,
                        atol: 1e-9,
                    },
                    t_end,
                ),
            ),
            (
                "OperatorSplitting",
                Box::new(splitting),
                config(n, IntegratorKind::Imex, fixed(explicit), t_end),
            ),
        ]
    }

    fn field(value: &ContextValue) -> &DVector<f64> {
        value.as_scalar_field().unwrap()
    }

    // ── Saved states ──────────────────────────────────────────────────────────

    #[test]
    fn every_saved_state_holds_a_declared_constant_value() {
        let n = 21;
        let bc = || PinnedNodes::new(&[0, 20], |_| 1.25).declared();
        for (name, solver, config) in solvers(n, &bc) {
            let scenario = Scenario::multi(vec![heat_domain(n, ALPHA, bc())]).unwrap();
            let result = solver.solve(&scenario, &config).unwrap();
            assert!(result.states.len() > 2, "{name}");
            for (k, state) in result.states.iter().enumerate() {
                let u = field(state);
                assert_eq!((u[0], u[n - 1]), (1.25, 1.25), "{name}, saved state {k}");
            }
        }
    }

    #[test]
    fn every_saved_state_holds_a_time_dependent_value_at_its_time() {
        let n = 21;
        // Differs from the initial value 1 at the ends: the initial state
        // must be corrected too.
        let g = |t: f64| 1.5 + 0.5 * t;
        let bc = move || PinnedNodes::new(&[0, 20], g).declared();
        for (name, solver, config) in solvers(n, &bc) {
            let scenario = Scenario::multi(vec![heat_domain(n, ALPHA, bc())]).unwrap();
            let result = solver.solve(&scenario, &config).unwrap();
            for (state, &t) in result.states.iter().zip(&result.times) {
                let u = field(state);
                assert_eq!(u[0].to_bits(), g(t).to_bits(), "{name}, t = {t}");
                assert_eq!(u[n - 1].to_bits(), g(t).to_bits(), "{name}, t = {t}");
            }
        }
    }

    #[test]
    fn the_orchestrator_saves_states_holding_the_conditions() {
        let n = 21;
        // Differs from the initial value 1 at the ends: the initial state
        // must be corrected too.
        let g = |t: f64| 1.5 + 0.5 * t;
        let scenario = Scenario::multi(vec![heat_domain(
            n,
            ALPHA,
            PinnedNodes::new(&[0, 20], g).declared(),
        )])
        .unwrap();
        let config = OrchestratorConfig::new(TimeConfiguration::new(
            0.1,
            StepControl::Fixed { dt: explicit_dt(n) },
        ))
        .with_calculator(laplacian_calculator(n));
        let quantity = PhysicalQuantity::concentration();
        let id = DomainId::new("bar");
        let result = MultiDomainOrchestrator::new()
            .with_domain(
                id.clone(),
                Box::new(crate::solver::methods::ForwardEulerSolver),
                quantity.clone(),
            )
            .run(&scenario, &config)
            .unwrap();
        for (state, &t) in result.states.iter().zip(&result.times) {
            let u = field(state.get(&id, &quantity).unwrap());
            assert_eq!(
                (u[0].to_bits(), u[n - 1].to_bits()),
                (g(t).to_bits(), g(t).to_bits()),
                "t = {t}"
            );
        }
    }

    // ── Spatial order ─────────────────────────────────────────────────────────

    /// Largest interior error against the exact solution, at the last saved time.
    fn interior_error(n: usize, solver: &dyn Solver, config: &SolverConfiguration) -> f64 {
        let bc = PinnedNodes::new(&[0, n - 1], |_| 1.0).declared();
        let scenario = Scenario::multi(vec![heat_domain(n, ALPHA, bc)]).unwrap();
        let result = solver.solve(&scenario, config).unwrap();
        let u = field(result.states.last().unwrap());
        let t = *result.times.last().unwrap();
        let mesh = grid(n);
        (1..n - 1)
            .map(|i| (u[i] - exact(mesh.coordinates(i)[0], t)).abs())
            .fold(0.0, f64::max)
    }

    /// One of the solvers whose spatial order is checked, on `n` nodes:
    /// explicit methods at a fifth of the diffusion limit, implicit ones at
    /// a step small enough for the spatial error to dominate.
    fn order_case(name: &str, n: usize) -> (Box<dyn Solver>, SolverConfiguration) {
        use crate::solver::methods::{
            BDF2Solver, CrankNicolsonSolver, ForwardEulerSolver, RK4Solver,
        };
        let t_end = 0.2;
        let explicit = StepControl::Fixed { dt: explicit_dt(n) };
        let implicit = StepControl::Fixed { dt: 1e-3 };
        match name {
            "Euler" => (
                Box::new(ForwardEulerSolver),
                config(n, IntegratorKind::Euler, explicit, t_end),
            ),
            "RK4" => (
                Box::new(RK4Solver),
                config(n, IntegratorKind::RK4, explicit, t_end),
            ),
            "CrankNicolson" => (
                Box::new(CrankNicolsonSolver::new()),
                config(n, IntegratorKind::CrankNicolson, implicit, t_end),
            ),
            "BDF2" => (
                Box::new(BDF2Solver::new()),
                config(n, IntegratorKind::BDF2, implicit, t_end),
            ),
            _ => unreachable!("no order case named {name}"),
        }
    }

    #[test]
    fn held_ends_keep_second_order_in_space() {
        for name in ["Euler", "RK4", "CrankNicolson", "BDF2"] {
            let errors: Vec<f64> = [11, 21, 41]
                .iter()
                .map(|&n| {
                    let (solver, config) = order_case(name, n);
                    interior_error(n, solver.as_ref(), &config)
                })
                .collect();
            for pair in errors.windows(2) {
                let ratio = pair[0] / pair[1];
                assert!(
                    (3.8..=4.2).contains(&ratio),
                    "{name}: errors {errors:?}, ratio {ratio}"
                );
            }
        }
    }

    // ── Evaluation order and held derivative ─────────────────────────────────

    fn evaluate(
        n: usize,
        bc: PinnedNodes,
        state: &mut ContextValue,
    ) -> Result<ContextValue, OxiflowError> {
        let domain = heat_domain(n, ALPHA, bc);
        let calc = laplacian_calculator(n);
        let chain: Vec<&dyn ContextCalculator> = vec![calc.as_ref()];
        evaluate_derivative(&domain, &chain, state, 0.0, 0.01)
    }

    #[test]
    fn calculators_see_the_constrained_state() {
        let n = 11;
        let initial = Heat { alpha: ALPHA }.initial_state(&grid(n));
        let mut drifted = initial.clone();
        if let ContextValue::ScalarField(u) = &mut drifted {
            u[0] = 5.0;
        }
        let mut reference = initial.clone();
        let expected = evaluate(n, PinnedNodes::new(&[0], |_| 1.0), &mut reference).unwrap();
        let derivative = evaluate(n, PinnedNodes::new(&[0], |_| 1.0), &mut drifted).unwrap();
        assert_eq!(field(&drifted)[0], 1.0);
        assert_eq!(
            field(&derivative)[1].to_bits(),
            field(&expected)[1].to_bits()
        );
    }

    #[test]
    fn only_declared_nodes_get_a_zero_derivative() {
        let n = 11;
        let initial = Heat { alpha: ALPHA }.initial_state(&grid(n));
        let undeclared =
            evaluate(n, PinnedNodes::new(&[0, 10], |_| 1.0), &mut initial.clone()).unwrap();
        let declared = evaluate(
            n,
            PinnedNodes::new(&[0, 10], |_| 1.0).declared(),
            &mut initial.clone(),
        )
        .unwrap();
        assert_ne!(field(&undeclared)[0], 0.0);
        assert_eq!((field(&declared)[0], field(&declared)[10]), (0.0, 0.0));
        for i in 1..n - 1 {
            assert_eq!(
                field(&declared)[i].to_bits(),
                field(&undeclared)[i].to_bits(),
                "node {i}"
            );
        }
    }

    #[test]
    fn a_declared_node_outside_the_state_is_refused() {
        let n = 11;
        let mut state = Heat { alpha: ALPHA }.initial_state(&grid(n));
        let result = evaluate(
            n,
            PinnedNodes::new(&[0], |_| 1.0).declaring(&[n]),
            &mut state,
        );
        assert!(
            matches!(
                result,
                Err(OxiflowError::PreconditionFailed {
                    context: "BoundaryCondition::constrained_nodes",
                    ..
                })
            ),
            "{result:?}"
        );
    }

    // ── Calculators required by a condition ───────────────────────────────────

    /// Sets node 0 to the scalar `inlet` read in the context.
    #[derive(Debug)]
    struct InletFromContext;

    impl RequiresContext for InletFromContext {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::External {
                name: "inlet".into(),
            }]
        }
    }

    impl BoundaryCondition for InletFromContext {
        fn boundary_type(&self) -> BoundaryType {
            BoundaryType::Dirichlet
        }
        fn apply(
            &self,
            state: &mut DVector<f64>,
            ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            state[0] = ctx.scalar(ContextVariable::External {
                name: "inlet".into(),
            })?;
            Ok(())
        }
        fn constrained_nodes(&self, _mesh: &dyn Mesh) -> Vec<usize> {
            vec![0]
        }
    }

    /// Provides `name` as `factor` times the scalar `source` (or `factor`
    /// alone without source), declaring `source` in `depends_on()`.
    #[derive(Debug)]
    struct Scaled {
        name: &'static str,
        source: Option<&'static str>,
        factor: f64,
    }

    impl RequiresContext for Scaled {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
        fn depends_on(&self) -> Vec<ContextVariable> {
            self.source
                .map(|s| ContextVariable::External { name: s.into() })
                .into_iter()
                .collect()
        }
    }

    impl ContextCalculator for Scaled {
        fn provides(&self) -> ContextVariable {
            ContextVariable::External {
                name: self.name.into(),
            }
        }
        fn compute(
            &self,
            _state: &ContextValue,
            ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let base = match self.source {
                Some(s) => ctx.scalar(ContextVariable::External { name: s.into() })?,
                None => 1.0,
            };
            Ok(ContextValue::Scalar(self.factor * base))
        }
    }

    #[test]
    fn calculators_required_by_a_condition_run_first_with_their_dependencies() {
        let n = 11;
        let unrelated = Scaled {
            name: "unrelated",
            source: None,
            factor: 7.0,
        };
        let base = Scaled {
            name: "base",
            source: None,
            factor: 3.0,
        };
        let inlet = Scaled {
            name: "inlet",
            source: Some("base"),
            factor: 0.5,
        };
        let lap = laplacian_calculator(n);
        let chain: Vec<&dyn ContextCalculator> = vec![&unrelated, &base, &inlet, lap.as_ref()];

        let domain = Domain::new("bar", Box::new(Heat { alpha: ALPHA }), Box::new(grid(n)))
            .with_boundary_conditions(vec![Box::new(InletFromContext)]);
        assert_eq!(
            boundary_calculators(&domain, &chain),
            vec![false, true, true, false]
        );

        let mut state = Heat { alpha: ALPHA }.initial_state(&grid(n));
        let derivative = evaluate_derivative(&domain, &chain, &mut state, 0.0, 0.01).unwrap();
        assert_eq!(field(&state)[0], 1.5);
        assert_eq!(field(&derivative)[0], 0.0);
    }
}
