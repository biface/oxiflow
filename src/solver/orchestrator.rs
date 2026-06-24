//! # Module `solver::orchestrator`
//!
//! Multi-domain orchestration — DD-031.
//!
//! ## Why this exists
//!
//! [`Solver`](crate::solver::Solver) is deliberately single-domain — it
//! drives one [`Domain`](crate::solver::scenario::Domain) through a full
//! time range. Coupled scenarios
//! (lahar–lake, #40) need several domains advancing together, each
//! exchanging state with the others via
//! [`CouplingOperator`](crate::coupling::CouplingOperator) (INV-3,
//! DD-011) between steps.
//!
//! Rather than generalising `Solver`/`SimulationResult` for this — which
//! would force every coupled domain through the same integrator —
//! [`MultiDomainOrchestrator`] drives one [`SteppableSolver`] *per domain*,
//! so the lahar domain can run `ForwardEulerSolver` while the lake domain
//! runs `RK4Solver`, or any other combination.
//!
//! ## Scope (v1)
//!
//! `dt` is synchronised across all domains: every domain advances by the
//! same step before couplings are applied. Per-domain `dt` (multirate /
//! sub-cycling) is a substantially harder problem — interface time
//! interpolation, coupling stability — deliberately deferred until a
//! concrete case requires it (see DD-031).
//!
//! ## Per-step order
//!
//! At each synchronised step:
//!
//! 1. **Advance every domain** by one step, each via its own registered
//!    `SteppableSolver` (which itself follows the contractual
//!    calculators → boundary conditions → `compute_physics` order, see
//!    [`crate::solver`]).
//! 2. **Apply every registered `CouplingOperator`**, in declaration order,
//!    each reading the just-updated [`MultiDomainState`] and returning an
//!    updated one.
//! 3. **Guard against divergence** across all domains.

use std::collections::HashMap;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::quantity::PhysicalQuantity;
use crate::context::state::MultiDomainState;
use crate::context::{ContextCalculator, ContextValue};
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::{StepControl, TimeConfiguration};
use crate::solver::methods::{check_finite, SteppableSolver};
use crate::solver::scenario::{DomainId, Scenario};

// ── OrchestratorConfig ──────────────────────────────────────────────────────────

/// Configuration shared by every domain in an orchestrated run.
///
/// Mirrors [`SolverConfiguration`](crate::solver::config::SolverConfiguration)
/// minus `integrator` — integrator choice is per-domain here, via
/// [`MultiDomainOrchestrator::with_domain`].
#[non_exhaustive]
#[derive(Debug)]
pub struct OrchestratorConfig {
    /// Temporal parameters — t_end, step control, save frequency. Only
    /// `StepControl::Fixed` is supported (DD-031 v1: synchronised `dt`).
    pub time: TimeConfiguration,
    /// Context variable calculators, shared across all domains — built
    /// once from `scenario.context_requirements()`, which already
    /// aggregates every domain's needs.
    pub calculators: Vec<Box<dyn ContextCalculator>>,
}

impl OrchestratorConfig {
    /// Creates a new configuration with no calculators.
    pub fn new(time: TimeConfiguration) -> Self {
        Self {
            time,
            calculators: Vec::new(),
        }
    }

    /// Adds a context calculator (builder pattern).
    pub fn with_calculator(mut self, calc: Box<dyn ContextCalculator>) -> Self {
        self.calculators.push(calc);
        self
    }
}

// ── MultiDomainSimulationResult ──────────────────────────────────────────────────

/// Result of a completed multi-domain orchestrated run.
///
/// Distinct from [`SimulationResult`](crate::solver::SimulationResult) —
/// one entry per saved time, each holding every domain's state rather than
/// a single domain's `ContextValue`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MultiDomainSimulationResult {
    /// Saved multi-domain states at each recorded time.
    pub states: Vec<MultiDomainState>,
    /// Simulation times corresponding to each saved state.
    pub times: Vec<f64>,
    /// Total number of synchronised steps taken (may be larger than
    /// `states.len()` if `save_every > 1`).
    pub n_steps: usize,
}

impl MultiDomainSimulationResult {
    /// Returns the number of saved states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` if no states were saved.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns the final simulation time.
    pub fn t_final(&self) -> Option<f64> {
        self.times.last().copied()
    }
}

// ── MultiDomainOrchestrator ───────────────────────────────────────────────────────

/// Drives multiple coupled [`Domain`](crate::solver::scenario::Domain)s,
/// each with its own [`SteppableSolver`].
///
/// See [module documentation](self) for the per-step order and the v1
/// synchronised-`dt` scope.
#[derive(Default)]
pub struct MultiDomainOrchestrator {
    solvers: HashMap<DomainId, Box<dyn SteppableSolver>>,
    /// The `PhysicalQuantity` each domain's primary state is keyed under
    /// in `MultiDomainState` — convention established by
    /// `tests/coupling_proto.rs` (v0.3.0): the caller picks one explicitly,
    /// `PhysicalModel` does not declare it itself.
    quantities: HashMap<DomainId, PhysicalQuantity>,
}

impl MultiDomainOrchestrator {
    /// Creates an empty orchestrator with no domains registered.
    pub fn new() -> Self {
        Self {
            solvers: HashMap::new(),
            quantities: HashMap::new(),
        }
    }

    /// Registers the solver and state quantity key to use for `domain_id`.
    ///
    /// Every domain present in the `Scenario` passed to [`run`](Self::run)
    /// must have a corresponding entry, or `run` returns
    /// `OxiflowError::InvalidDomain`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let orchestrator = MultiDomainOrchestrator::new()
    ///     .with_domain(lahar_id, Box::new(ForwardEulerSolver), PhysicalQuantity::concentration())
    ///     .with_domain(lake_id, Box::new(RK4Solver), PhysicalQuantity::concentration());
    /// ```
    pub fn with_domain(
        mut self,
        domain_id: DomainId,
        solver: Box<dyn SteppableSolver>,
        quantity: PhysicalQuantity,
    ) -> Self {
        self.solvers.insert(domain_id.clone(), solver);
        self.quantities.insert(domain_id, quantity);
        self
    }

    /// Runs the orchestrated simulation and returns the collected states.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::InvalidDomain` if any domain in `scenario` has no
    ///   registered solver/quantity, if `scenario` has no domains, or if
    ///   `dt`/`t_end` are invalid.
    /// - `OxiflowError::InvalidDomain` if `config.time.step_control` is not
    ///   `StepControl::Fixed` (DD-031 v1 scope).
    pub fn run(
        &self,
        scenario: &Scenario,
        config: &OrchestratorConfig,
    ) -> Result<MultiDomainSimulationResult, OxiflowError> {
        scenario.validate()?;

        if scenario.n_domains() == 0 {
            return Err(OxiflowError::InvalidDomain(
                "scenario has no domains".into(),
            ));
        }
        for domain in scenario.domains() {
            if !self.solvers.contains_key(&domain.id) {
                return Err(OxiflowError::InvalidDomain(format!(
                    "no SteppableSolver registered for domain '{}' -- call with_domain() for it",
                    domain.id
                )));
            }
        }

        let dt = match &config.time.step_control {
            StepControl::Fixed { dt } => *dt,
            _ => {
                return Err(OxiflowError::InvalidDomain(
                    "MultiDomainOrchestrator only supports StepControl::Fixed \
                     (DD-031 v1: dt synchronised across all domains)"
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

        // One shared calculator chain, built from the aggregated
        // requirements of every domain + coupling — same convention
        // `Solver::solve()` uses for the single-domain case.
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &config.calculators)?;

        let n_steps = ((t_end - t_start) / dt).round() as usize;
        let save_every = config.time.save_every.unwrap_or(1);
        let capacity = n_steps / save_every + 1;

        // ── Initial state — one entry per domain ────────────────────────────
        let mut multi_state = MultiDomainState::new();
        for domain in scenario.domains() {
            let quantity = self.quantity_for(&domain.id)?;
            let initial = domain.model.initial_state(domain.mesh.as_ref());
            multi_state.set(domain.id.clone(), quantity, initial);
        }

        // Per-domain history buffers, sized to each domain's own solver
        // (DD-034) — empty for one-step methods (history_depth() == 0,
        // the default), populated for multi-step ones (BDF2: depth 1).
        let mut histories: HashMap<DomainId, Vec<ContextValue>> = scenario
            .domains()
            .iter()
            .map(|d| (d.id.clone(), Vec::new()))
            .collect();

        let mut states: Vec<MultiDomainState> = Vec::with_capacity(capacity);
        let mut times: Vec<f64> = Vec::with_capacity(capacity);
        states.push(multi_state.clone());
        times.push(t_start);

        // ── Time loop ────────────────────────────────────────────────────────
        for step in 0..n_steps {
            let t = t_start + (step as f64) * dt;
            let t_next = t_start + ((step + 1) as f64) * dt;

            // 1. Advance every domain by one step, each with its own solver.
            for domain in scenario.domains() {
                let quantity = self.quantity_for(&domain.id)?;
                let solver = &self.solvers[&domain.id];

                let mut state = multi_state
                    .get(&domain.id, &quantity)
                    .ok_or_else(|| {
                        OxiflowError::InvalidDomain(format!(
                            "domain '{}' has no state for its registered quantity",
                            domain.id
                        ))
                    })?
                    .clone();

                let next_state = {
                    let history = &histories[&domain.id];
                    solver.step(domain, &chain, &mut state, history, t, dt)?
                };

                // `state` was mutated in-place by BC application inside
                // `step()` above -- push *that* corrected u^n into history
                // (depth-capped per the solver's own declared need), not
                // the pre-correction value.
                let depth = solver.history_depth();
                if depth > 0 {
                    let hist = histories.get_mut(&domain.id).unwrap();
                    hist.insert(0, state);
                    hist.truncate(depth);
                }

                multi_state.set(domain.id.clone(), quantity, next_state);
            }

            // 2. Exchange state across every coupling interface, in
            //    declaration order -- each operator sees the result of the
            //    previous one.
            let ctx = ComputeContext::new(t_next, dt);
            for coupling in scenario.couplings() {
                multi_state = coupling.apply(&multi_state, &ctx, coupling.interface())?;
            }

            // 3. Guard against divergence across all domains.
            for domain in scenario.domains() {
                let quantity = self.quantity_for(&domain.id)?;
                if let Some(value) = multi_state.get(&domain.id, &quantity) {
                    check_finite(value, t_next)?;
                }
            }

            if (step + 1) % save_every == 0 {
                states.push(multi_state.clone());
                times.push(t_next);
            }
        }

        Ok(MultiDomainSimulationResult {
            states,
            times,
            n_steps,
        })
    }

    /// Looks up the registered quantity for `domain_id`.
    fn quantity_for(&self, domain_id: &DomainId) -> Result<PhysicalQuantity, OxiflowError> {
        self.quantities.get(domain_id).cloned().ok_or_else(|| {
            OxiflowError::InvalidDomain(format!(
                "no PhysicalQuantity key registered for domain '{domain_id}' -- call with_domain() for it"
            ))
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::value::ContextValue;
    use crate::context::variable::ContextVariable;
    use crate::coupling::{CouplingOperator, Interface};
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::methods::{ForwardEulerSolver, RK4Solver};
    use crate::solver::scenario::Domain;
    use nalgebra::DVector;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    /// Exponential decay: du/dt = -lambda * u.
    #[derive(Debug)]
    struct DecayModel {
        lambda: f64,
    }

    impl RequiresContext for DecayModel {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for DecayModel {
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
            "decay"
        }
    }

    /// Passive receiver: du/dt = 0.
    #[derive(Debug)]
    struct PassiveModel;

    impl RequiresContext for PassiveModel {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for PassiveModel {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(DVector::zeros(u.len())))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::zeros(mesh.n_dof()))
        }

        fn name(&self) -> &str {
            "passive"
        }
    }

    /// Transfers a fixed fraction of the source domain's field to the
    /// target domain at every call -- counts invocations for assertions.
    #[derive(Debug)]
    struct CountingMassTransfer {
        alpha: f64,
        interface: Interface,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl RequiresContext for CountingMassTransfer {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl CouplingOperator for CountingMassTransfer {
        fn apply(
            &self,
            states: &MultiDomainState,
            _ctx: &ComputeContext,
            interface: &Interface,
        ) -> Result<MultiDomainState, OxiflowError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            let quantity = PhysicalQuantity::concentration();
            let source = states
                .get(interface.source(), &quantity)
                .ok_or_else(|| OxiflowError::InvalidDomain("missing source field".into()))?
                .as_scalar_field()?;
            let transferred = source.map(|v| self.alpha * v);

            let mut result = states.clone();
            result.set(
                interface.target().clone(),
                quantity,
                ContextValue::ScalarField(transferred),
            );
            Ok(result)
        }

        fn interface(&self) -> &Interface {
            &self.interface
        }
    }

    fn source_id() -> DomainId {
        DomainId::new("source")
    }
    fn target_id() -> DomainId {
        DomainId::new("target")
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    fn make_scenario() -> (Scenario, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let source = Domain::new(
            source_id(),
            Box::new(DecayModel { lambda: 0.5 }),
            make_mesh(3),
        );
        let target = Domain::new(target_id(), Box::new(PassiveModel), make_mesh(3));
        let interface = Interface::new(source_id(), target_id());
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let coupling = Box::new(CountingMassTransfer {
            alpha: 0.1,
            interface,
            calls: calls.clone(),
        });
        let scenario = Scenario::multi(vec![source, target])
            .unwrap()
            .with_coupling(coupling);
        (scenario, calls)
    }

    fn make_config(t_end: f64, dt: f64) -> OrchestratorConfig {
        OrchestratorConfig::new(TimeConfiguration::new(t_end, StepControl::Fixed { dt }))
    }

    // ── Basic execution ────────────────────────────────────────────────────────

    #[test]
    fn run_end_to_end_two_domains() {
        let (scenario, _calls) = make_scenario();
        let orchestrator = MultiDomainOrchestrator::new()
            .with_domain(
                source_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            )
            .with_domain(
                target_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            );

        let result = orchestrator.run(&scenario, &make_config(1.0, 0.1)).unwrap();
        assert_eq!(result.n_steps, 10);
        assert!(!result.is_empty());
    }

    #[test]
    fn missing_solver_registration_returns_error() {
        let (scenario, _calls) = make_scenario();
        // Only "source" registered -- "target" is missing.
        let orchestrator = MultiDomainOrchestrator::new().with_domain(
            source_id(),
            Box::new(ForwardEulerSolver),
            PhysicalQuantity::concentration(),
        );

        let err = orchestrator
            .run(&scenario, &make_config(1.0, 0.1))
            .unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    #[test]
    fn adaptive_step_control_returns_error() {
        let (scenario, _calls) = make_scenario();
        let orchestrator = MultiDomainOrchestrator::new()
            .with_domain(
                source_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            )
            .with_domain(
                target_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            );

        let config = OrchestratorConfig::new(TimeConfiguration::new(
            1.0,
            StepControl::Adaptive {
                dt_init: 0.1,
                dt_min: 1e-6,
                dt_max: 1.0,
                rtol: 1e-6,
                atol: 1e-9,
            },
        ));

        let err = orchestrator.run(&scenario, &config).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    // ── Coupling invocation ─────────────────────────────────────────────────────

    #[test]
    fn coupling_invoked_exactly_once_per_step() {
        let (scenario, calls) = make_scenario();
        let orchestrator = MultiDomainOrchestrator::new()
            .with_domain(
                source_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            )
            .with_domain(
                target_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            );

        let result = orchestrator.run(&scenario, &make_config(0.5, 0.1)).unwrap();
        assert_eq!(result.n_steps, 5);

        // 5 steps -> exactly 5 invocations, not 0 (never called) and not
        // more (e.g. accidentally invoked once per domain per step).
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 5);

        // Cross-check via effect: target received non-zero mass, which
        // only the coupling (never PassiveModel itself, zero derivative)
        // can produce.
        let final_state = result.states.last().unwrap();
        let target_field = final_state
            .get(&target_id(), &PhysicalQuantity::concentration())
            .unwrap()
            .as_scalar_field()
            .unwrap();
        assert!(target_field.iter().all(|v| *v > 0.0));
    }

    // ── Mixed integrators per domain ─────────────────────────────────────────────

    #[test]
    fn domains_may_use_different_integrators() {
        let (scenario, _calls) = make_scenario();
        let orchestrator = MultiDomainOrchestrator::new()
            .with_domain(
                source_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            )
            .with_domain(
                target_id(),
                Box::new(RK4Solver),
                PhysicalQuantity::concentration(),
            );

        let result = orchestrator.run(&scenario, &make_config(1.0, 0.1)).unwrap();
        assert_eq!(result.n_steps, 10);
        assert!(!result.is_empty());
    }

    // ── Result accessors ────────────────────────────────────────────────────────

    #[test]
    fn result_t_final() {
        let (scenario, _calls) = make_scenario();
        let orchestrator = MultiDomainOrchestrator::new()
            .with_domain(
                source_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            )
            .with_domain(
                target_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            );

        let result = orchestrator.run(&scenario, &make_config(1.0, 0.1)).unwrap();
        assert!((result.t_final().unwrap() - 1.0).abs() < 1e-9);
    }
}
