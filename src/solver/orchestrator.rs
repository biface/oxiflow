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
//!    [`crate::solver`]). Split into two passes internally (DD-048
//!    consequence): a Jacobi-parallel compute pass (each domain reads
//!    only the pre-step state, independent by construction, optionally
//!    dispatched through Rayon above [`MultiDomainOrchestrator::parallel_threshold`])
//!    followed by a sequential apply pass that writes the results back
//!    into [`MultiDomainState`]/history. See
//!    [`MultiDomainOrchestrator::compute_domain_steps`] for why the split
//!    exists and what it does and does not parallelize.
//! 2. **Apply every registered `CouplingOperator`**, in declaration order,
//!    each reading the just-updated [`MultiDomainState`] and returning an
//!    updated one. Always sequential, and always after every domain's
//!    Phase-1 write has completed (INV-3).
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
#[cfg(feature = "parallel")]
use crate::solver::parallel::ParallelThreshold;
use crate::solver::scenario::{Domain, DomainId, Scenario};

/// Default Phase-1 dispatch threshold — a back-of-envelope estimate, not
/// a measured crossover point like
/// [`crate::operators::fd::default_parallel_threshold`] (49_999).
///
/// The two sites dispatch at a different grain, so the numbers are not
/// comparable directly:
///
/// - `fd.rs` splits `n` elements finely (`into_par_iter()` over a
///   `Range`), each costing a few flops (~1 ns) -- amortizing Rayon's
///   per-chunk overhead needs a very large `n`.
/// - Phase 1 dispatches over *domains* (`.par_iter()`) -- typically a
///   handful, rarely more -- where each task is an entire
///   `solver.step()` call (calculator chain + boundary conditions +
///   physics evaluation, possibly several stages for RK4/BDF2). Few,
///   large tasks amortize Rayon's fixed join/dispatch overhead (order of
///   1-few µs) at a much lower total operation count than fine-grained
///   dispatch does.
///
/// Rough estimate: a full step costs perhaps 10-50x a bare stencil
/// evaluation per degree of freedom (~10-50 ns/dof instead of ~1 ns).
/// With a healthy safety margin (~10x) over Rayon's fixed overhead, the
/// aggregate mesh volume needs to reach on the order of 10-50 µs worth
/// of work, i.e. roughly 1,000-5,000 dof. `5_000` is picked from that
/// range -- **not measured**, unlike `fd.rs`'s 49_999. Replace it with a
/// real value from
/// [`calibrate_parallel_threshold`](crate::solver::parallel::calibrate_parallel_threshold)
/// (representative closure: a full `solver.step()` call, not a
/// placeholder) via [`MultiDomainOrchestrator::with_parallel_threshold`]
/// or [`MultiDomainOrchestrator::set_parallel_threshold`] once a real
/// workload profile is available.
#[cfg(feature = "parallel")]
pub(crate) fn default_parallel_threshold() -> usize {
    5_000
}

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
    /// Phase-1 dispatch threshold (DD-048 consequence) — gates the
    /// "advance every domain" pass between a sequential loop and a Rayon
    /// dispatch across domains. Measured against the *aggregate* mesh
    /// volume (sum of every domain's [`Mesh::n_dof`](crate::mesh::Mesh::n_dof)
    /// in the scenario being run), not the number of domains — domain
    /// count alone is blind to workload: two domains can mean two heavy
    /// meshes or two trivial ones, and `n_dof()` already exists on every
    /// domain's mesh with no new API needed.
    ///
    /// Known limitation, accepted deliberately (DD-048/P5 does not
    /// require an exact cost measure, only a measured one): this
    /// approximation does not weight for per-node cost differences
    /// between solver types — an adaptive integrator costs more per node
    /// than [`ForwardEulerSolver`](crate::solver::methods::ForwardEulerSolver),
    /// and the aggregate `n_dof()` sum cannot see that difference.
    #[cfg(feature = "parallel")]
    parallel_threshold: ParallelThreshold,
}

impl MultiDomainOrchestrator {
    /// Creates an empty orchestrator with no domains registered.
    pub fn new() -> Self {
        Self {
            solvers: HashMap::new(),
            quantities: HashMap::new(),
            #[cfg(feature = "parallel")]
            parallel_threshold: ParallelThreshold::unset(),
        }
    }

    /// Overrides the Phase-1 dispatch threshold (builder style).
    ///
    /// See the [`parallel_threshold`](Self::parallel_threshold) field docs
    /// for what this threshold measures (aggregate mesh volume, not
    /// domain count) and its known limitation. Only exists when the crate
    /// is built with the `parallel` feature.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — see
    /// [`ParallelThreshold::set`](crate::solver::parallel::ParallelThreshold::set).
    #[cfg(feature = "parallel")]
    pub fn with_parallel_threshold(self, threshold: usize) -> Self {
        self.parallel_threshold.set(threshold);
        self
    }

    /// Overrides the Phase-1 dispatch threshold at runtime, through a
    /// shared reference — no rebuild required.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` (see
    /// [`ParallelThreshold::set`](crate::solver::parallel::ParallelThreshold::set)).
    #[cfg(feature = "parallel")]
    pub fn set_parallel_threshold(&self, threshold: usize) {
        self.parallel_threshold.set(threshold);
    }

    /// Returns the configured Phase-1 dispatch threshold, or
    /// [`default_parallel_threshold`] (`5_000` — a rough estimate, not
    /// measured; see its own docs) if none was set.
    #[cfg(feature = "parallel")]
    pub fn parallel_threshold(&self) -> usize {
        self.parallel_threshold.get(default_parallel_threshold())
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

            // 1. Advance every domain by one step, each with its own
            //    solver — two-phase Jacobi split (issue tracking this
            //    consequence of DD-048/#140).
            //
            // Domains only read the pre-step (t^n) `multi_state` and never
            // see another domain's post-step value within the same
            // synchronised step, so the *values* computed below are
            // already independent (Jacobi, confirmed prior session). What
            // blocked a direct Rayon dispatch was the loop's shape, not
            // the numerics: `multi_state.set(...)` and `histories` were
            // both mutated in place, per domain, inside the same
            // iteration -- concurrent `&mut` access to either shared
            // structure is neither safe nor something Rust's aliasing
            // rules allow without a lock.
            //
            // Phase 1a computes every domain's result into its own `Vec`
            // slot -- no shared mutation during this pass, safe to
            // dispatch in parallel once the aggregate mesh volume crosses
            // `parallel_threshold()`. Phase 1b applies the collected
            // results to `multi_state`/`histories` sequentially -- cheap,
            // O(domain count), not the numerical work itself.
            let results: Vec<DomainStepResult> =
                self.compute_domain_steps(scenario, &multi_state, &histories, &chain, t, dt)?;

            for result in results {
                if result.history_depth > 0 {
                    if let Some(entry) = result.history_entry {
                        let hist = histories.get_mut(&result.domain_id).unwrap();
                        hist.insert(0, entry);
                        hist.truncate(result.history_depth);
                    }
                }
                multi_state.set(result.domain_id, result.quantity, result.next_state);
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

    /// Phase 1a: computes every domain's next state, independently.
    ///
    /// Reads only the pre-step `multi_state`/`histories` (both taken by
    /// shared reference) and writes nothing -- results are collected into
    /// a `Vec`, one domain-local slot each, for [`run`](Self::run)'s
    /// Phase 1b to apply sequentially. Dispatched through Rayon once the
    /// aggregate mesh volume (`Σ domain.mesh.n_dof()`) reaches
    /// [`parallel_threshold`](Self::parallel_threshold); a plain
    /// sequential loop otherwise, and always without the `parallel`
    /// feature.
    ///
    /// Errors from different domains may surface in a different order
    /// than the sequential path when run in parallel (Rayon's `collect`
    /// returns *an* error, not necessarily the first domain's in
    /// iteration order) -- domains are independent within a step, so
    /// which one's error is reported first does not change whether the
    /// step as a whole fails.
    fn compute_domain_steps(
        &self,
        scenario: &Scenario,
        multi_state: &MultiDomainState,
        histories: &HashMap<DomainId, Vec<ContextValue>>,
        chain: &[&dyn ContextCalculator],
        t: f64,
        dt: f64,
    ) -> Result<Vec<DomainStepResult>, OxiflowError> {
        let compute_one = |domain: &Domain| -> Result<DomainStepResult, OxiflowError> {
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
                solver.step(domain, chain, &mut state, history, t, dt)?
            };

            // `state` was mutated in-place by BC application inside
            // `step()` above -- carry *that* corrected u^n forward for
            // history (depth-capped per the solver's own declared need),
            // not the pre-correction value.
            let depth = solver.history_depth();
            let history_entry = if depth > 0 { Some(state) } else { None };

            Ok(DomainStepResult {
                domain_id: domain.id.clone(),
                quantity,
                next_state,
                history_entry,
                history_depth: depth,
            })
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let total_dof: usize = scenario.domains().iter().map(|d| d.mesh.n_dof()).sum();
            if total_dof >= self.parallel_threshold() {
                return scenario
                    .domains()
                    .par_iter()
                    .map(compute_one)
                    .collect::<Result<Vec<_>, _>>();
            }
        }

        scenario
            .domains()
            .iter()
            .map(compute_one)
            .collect::<Result<Vec<_>, _>>()
    }
}

/// One domain's Phase-1 result, collected by
/// [`MultiDomainOrchestrator::compute_domain_steps`] for the sequential
/// apply pass in [`MultiDomainOrchestrator::run`].
struct DomainStepResult {
    domain_id: DomainId,
    quantity: PhysicalQuantity,
    next_state: ContextValue,
    /// `Some(corrected_state)` when the solver's `history_depth() > 0` --
    /// the state `solver.step()` mutated in place via boundary-condition
    /// application, to push into `histories`.
    history_entry: Option<ContextValue>,
    history_depth: usize,
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

    // ── Phase 1 two-phase split (parallel feature) ────────────────────────────────

    /// Forcing the Rayon path (via a threshold of `1`, always crossed)
    /// must produce results identical to the sequential path -- the
    /// two-phase split changes *how* Phase 1 computes, never *what* it
    /// computes (Jacobi: domains are already independent within a step).
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_dispatch_matches_sequential_results() {
        let (scenario, _calls) = make_scenario();

        let sequential = MultiDomainOrchestrator::new()
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
        // Default threshold is `5_000` (see `default_parallel_threshold`)
        // -- this scenario's aggregate volume (2 domains x 3 dof = 6)
        // stays far below it, exercising the sequential path.
        let sequential_result = sequential.run(&scenario, &make_config(1.0, 0.1)).unwrap();

        let parallel = MultiDomainOrchestrator::new()
            .with_domain(
                source_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            )
            .with_domain(
                target_id(),
                Box::new(ForwardEulerSolver),
                PhysicalQuantity::concentration(),
            )
            .with_parallel_threshold(1);
        let parallel_result = parallel.run(&scenario, &make_config(1.0, 0.1)).unwrap();

        assert_eq!(sequential_result.n_steps, parallel_result.n_steps);
        assert_eq!(sequential_result.len(), parallel_result.len());

        let quantity = PhysicalQuantity::concentration();
        for (seq_state, par_state) in sequential_result
            .states
            .iter()
            .zip(parallel_result.states.iter())
        {
            for id in [source_id(), target_id()] {
                assert_eq!(
                    seq_state.get(&id, &quantity),
                    par_state.get(&id, &quantity),
                    "domain '{id}' diverged between sequential and parallel dispatch"
                );
            }
        }
    }

    /// A threshold of `1` is always crossed by any non-empty scenario --
    /// confirms the accessor round-trips through `with_parallel_threshold`
    /// rather than exercising dispatch behaviour (covered above).
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_threshold_accessor_reports_configured_value() {
        let orchestrator = MultiDomainOrchestrator::new().with_parallel_threshold(7);
        assert_eq!(orchestrator.parallel_threshold(), 7);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_threshold_defaults_to_five_thousand() {
        let orchestrator = MultiDomainOrchestrator::new();
        assert_eq!(orchestrator.parallel_threshold(), 5_000);
    }
}
