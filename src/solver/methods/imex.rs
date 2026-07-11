//! # Module `solver::methods::imex`
//!
//! Temporal operator splitting — `OperatorSplittingSolver` (DD-037, #45).
//!
//! ## Temporal coupling, not spatial
//!
//! This module composes **n ≥ 2** [`PhysicalModel`](crate::model::PhysicalModel)
//! evaluated in sequence on the **same state**, the **same mesh** — as
//! opposed to INV-3's spatial coupling (`CouplingOperator`, cross-domain,
//! asymmetric source → target at an `Interface`). See DD-037 for the full
//! distinction between the two families.
//!
//! oxiflow's canonical form already names two separate terms:
//!
//! $$\frac{\partial u}{\partial t} + \nabla \cdot F(u, \nabla u) = S(u, \mathbf{x}, t)$$
//!
//! Each [`SplitOperator`] carries **its own [`Domain`]** (same mesh, same
//! BCs where applicable, different model) rather than a plain model:
//! [`SteppableSolver::step`] always reads `domain.model`, so having each
//! contribution carry its own `Domain` is enough to isolate it correctly,
//! without touching `SteppableSolver` or any existing solver.
//!
//! ## v1 scope (#45)
//!
//! Only **one-step** sub-solvers (`history_depth() == 0`) are supported —
//! validated at construction. [`SplittingScheme::Strang`] is the only
//! implemented and tested variant; [`SplittingScheme::LieTrotter`] is
//! reserved (DD-037) and rejected at construction until a concrete case
//! requires it.
//!
//! ## The outer `Scenario`/`Domain`
//!
//! [`Solver::solve`] requires a `Scenario` — hence an outer `Domain` — even
//! though `OperatorSplittingSolver` doesn't need one to compute (it has
//! everything in its own operators). DD-037 settles this for
//! [`crate::model::CompositeModel`]: the outer `Domain` carries the sum of
//! the contributions, which makes `scenario.context_requirements()` and
//! `domain.model.initial_state()` correct without dedicated logic here —
//! and also serves as a testable monolithic reference run through an
//! ordinary solver (acceptance criterion 1 of #45).

use std::collections::HashMap;

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::StepControl;
use crate::solver::methods::{check_finite, SteppableSolver};
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// A named contribution to `∂u/∂t`, integrated by its own sub-solver.
///
/// Carries a full [`Domain`] (not just a model) — see the module docs for
/// why.
pub struct SplitOperator {
    /// Mesh, BCs, and model specific to this contribution.
    pub domain: Domain,
    /// Sub-solver integrating this contribution. Must have
    /// `history_depth() == 0` (v1 scope, DD-037).
    pub solver: Box<dyn SteppableSolver>,
}

/// Composition scheme for the operators over a step `dt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SplittingScheme {
    /// Full step of each operator, in sequence — order 1.
    ///
    /// Reserved (DD-037): same structure as `Strang`, but not implemented
    /// or tested here — `OperatorSplittingSolver::new` rejects it until a
    /// concrete case requires it.
    LieTrotter,
    /// Half-step on every operator but the last (full step), then passes
    /// back in reverse order — symmetric/palindromic composition, order 2.
    /// For n=2, this is exactly the scheme from #45: explicit half-step →
    /// full implicit step → explicit half-step.
    Strang,
}

/// Generic composite: n ≥ 2 operators sharing the same state, composed
/// according to a [`SplittingScheme`] (DD-037, #45).
pub struct OperatorSplittingSolver {
    operators: Vec<SplitOperator>,
    scheme: SplittingScheme,
}

/// Manual `Debug` — `Domain` (via `Box<dyn PhysicalModel>`/`Box<dyn Mesh>`)
/// has no `Debug` supertrait, so `#[derive(Debug)]` isn't available here.
/// Same proxy pattern as `FDGradientCalculator`'s `mesh_n_dof` field
/// (`context::calculators::spatial`): expose what's inspectable instead of
/// the trait object itself.
impl std::fmt::Debug for OperatorSplittingSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorSplittingSolver")
            .field("scheme", &self.scheme)
            .field(
                "operators",
                &self
                    .operators
                    .iter()
                    .map(|op| (op.domain.model.name().to_string(), op.domain.mesh.n_dof()))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl OperatorSplittingSolver {
    /// Builds a composite from at least two operators.
    ///
    /// # Errors
    ///
    /// `OxiflowError::PreconditionFailed` if:
    /// - fewer than two operators are provided;
    /// - a sub-solver has `history_depth() != 0` (v1 scope, DD-037);
    /// - `scheme` is `SplittingScheme::LieTrotter` (reserved, not
    ///   implemented — DD-037).
    pub fn new(
        operators: Vec<SplitOperator>,
        scheme: SplittingScheme,
    ) -> Result<Self, OxiflowError> {
        if operators.len() < 2 {
            return Err(OxiflowError::PreconditionFailed {
                context: "OperatorSplittingSolver::new",
                message: format!(
                    "at least two operators are required, got {}",
                    operators.len()
                ),
            });
        }

        if scheme == SplittingScheme::LieTrotter {
            return Err(OxiflowError::PreconditionFailed {
                context: "OperatorSplittingSolver::new",
                message: "SplittingScheme::LieTrotter is reserved (DD-037) — not yet \
                           implemented or tested; use SplittingScheme::Strang"
                    .into(),
            });
        }

        for op in &operators {
            let depth = op.solver.history_depth();
            if depth != 0 {
                return Err(OxiflowError::PreconditionFailed {
                    context: "OperatorSplittingSolver::new",
                    message: format!(
                        "sub-solver for operator '{}' has history_depth() == {depth} — \
                         only one-step sub-solvers (history_depth() == 0) are supported \
                         in v1 (DD-037)",
                        op.domain.model.name()
                    ),
                });
            }
        }

        Ok(Self { operators, scheme })
    }

    /// Ergonomic constructor for the n=2 case requested by #45: explicit
    /// half-step → full implicit step → explicit half-step.
    pub fn strang(
        domain_explicit: Domain,
        explicit_solver: Box<dyn SteppableSolver>,
        domain_implicit: Domain,
        implicit_solver: Box<dyn SteppableSolver>,
    ) -> Result<Self, OxiflowError> {
        Self::new(
            vec![
                SplitOperator {
                    domain: domain_explicit,
                    solver: explicit_solver,
                },
                SplitOperator {
                    domain: domain_implicit,
                    solver: implicit_solver,
                },
            ],
            SplittingScheme::Strang,
        )
    }

    /// Advances `state` by one outer step `dt`, according to `self.scheme`.
    fn apply_step(
        &self,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        match self.scheme {
            // Unreachable: rejected by `new` — kept for exhaustiveness in
            // case `SplittingScheme` gains more variants later (DD-037).
            SplittingScheme::LieTrotter => unreachable!(
                "SplittingScheme::LieTrotter is rejected by OperatorSplittingSolver::new"
            ),
            SplittingScheme::Strang => self.apply_strang(chain, state, t, dt),
        }
    }

    /// Symmetric/palindromic composition, generalised to n ≥ 2 (DD-037).
    ///
    /// Φ₁(dt/2) ∘ Φ₂(dt/2) ∘ … ∘ Φₙ₋₁(dt/2) ∘ Φₙ(dt) ∘ Φₙ₋₁(dt/2) ∘ … ∘ Φ₁(dt/2)
    ///
    /// For n=2: Φ₁ half-step (explicit) → Φ₂ full step (implicit) → Φ₁
    /// half-step — exactly the scheme from #45.
    fn apply_strang(
        &self,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        let n = self.operators.len();
        let half = dt / 2.0;

        let mut current = state.clone();
        let mut t_local = t;

        // Forward pass: half-step on operators[0..n-1].
        for op in &self.operators[..n - 1] {
            current = op
                .solver
                .step(&op.domain, chain, &mut current, &[], t_local, half)?;
            t_local += half;
        }

        // Full step on the last operator.
        let last = &self.operators[n - 1];
        current = last
            .solver
            .step(&last.domain, chain, &mut current, &[], t_local, dt)?;
        t_local += dt;

        // Return pass: half-step on operators[0..n-1], reverse order.
        for op in self.operators[..n - 1].iter().rev() {
            current = op
                .solver
                .step(&op.domain, chain, &mut current, &[], t_local, half)?;
            t_local += half;
        }

        Ok(current)
    }
}

impl Solver for OperatorSplittingSolver {
    /// Fixed-step loop — does not reuse
    /// [`SteppableSolver::solve_fixed_step`] (DD-035): this composite is not
    /// a `SteppableSolver` (same posture as DD-036 for DoPri45 — avoiding
    /// reopening multi-domain orchestration for a need #45 doesn't raise).
    /// The loop below deliberately mirrors `solve_fixed_step`'s shape to
    /// stay consistent with the other fixed-step solvers.
    fn solve(
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
                    "OperatorSplittingSolver only supports StepControl::Fixed (adaptive \
                     step control not supported)"
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

        let n_steps = ((t_end - t_start) / dt).round() as usize;
        let save_every = config.time.save_every.unwrap_or(1);
        let capacity = n_steps / save_every + 1;
        let mut states: Vec<ContextValue> = Vec::with_capacity(capacity);
        let mut times: Vec<f64> = Vec::with_capacity(capacity);

        states.push(u.clone());
        times.push(t_start);

        for step in 0..n_steps {
            let t = t_start + (step as f64) * dt;
            let t_next = t_start + ((step + 1) as f64) * dt;

            u = self.apply_step(&chain, &mut u, t, dt)?;

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::structured::UniformGrid1D;
    use crate::mesh::Mesh;
    use crate::model::{CompositeModel, PhysicalModel, RequiresContext};
    use crate::solver::config::{IntegratorKind, TimeConfiguration};
    use crate::solver::methods::euler::ForwardEulerSolver;
    use nalgebra::DVector;

    /// `du/dt = -rate * u` — pure decay, no time dependence.
    struct Decay {
        rate: f64,
    }
    impl RequiresContext for Decay {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }
    impl PhysicalModel for Decay {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(u.map(|v| -self.rate * v)))
        }
        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }
        fn name(&self) -> &str {
            "decay"
        }
    }

    fn make_mesh() -> UniformGrid1D {
        UniformGrid1D::new(5, 0.0, 1.0).unwrap()
    }

    fn make_domain(rate: f64) -> Domain {
        Domain::new("decay", Box::new(Decay { rate }), Box::new(make_mesh()))
    }

    fn make_solver() -> OperatorSplittingSolver {
        OperatorSplittingSolver::strang(
            make_domain(1.0),
            Box::new(ForwardEulerSolver),
            make_domain(2.0),
            Box::new(ForwardEulerSolver),
        )
        .unwrap()
    }

    #[test]
    fn rejects_fewer_than_two_operators() {
        let op = SplitOperator {
            domain: make_domain(1.0),
            solver: Box::new(ForwardEulerSolver),
        };
        let err = OperatorSplittingSolver::new(vec![op], SplittingScheme::Strang).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    #[test]
    fn rejects_lie_trotter_scheme() {
        let err = OperatorSplittingSolver::strang(
            make_domain(1.0),
            Box::new(ForwardEulerSolver),
            make_domain(2.0),
            Box::new(ForwardEulerSolver),
        );
        assert!(err.is_ok()); // strang() always uses Strang — sanity check
        let op_a = SplitOperator {
            domain: make_domain(1.0),
            solver: Box::new(ForwardEulerSolver),
        };
        let op_b = SplitOperator {
            domain: make_domain(2.0),
            solver: Box::new(ForwardEulerSolver),
        };
        let err = OperatorSplittingSolver::new(vec![op_a, op_b], SplittingScheme::LieTrotter)
            .unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    #[test]
    fn solve_reproduces_combined_decay_rate() {
        // Two separate decays (rate 1.0 + rate 2.0) split should
        // approximate the combined decay (rate 3.0) — acceptance
        // criterion 1 of #45.
        let solver = make_solver();

        let composite = CompositeModel::new(
            vec![
                Box::new(Decay { rate: 1.0 }) as Box<dyn PhysicalModel>,
                Box::new(Decay { rate: 2.0 }) as Box<dyn PhysicalModel>,
            ],
            "combined_decay",
        )
        .unwrap();
        let scenario = Scenario::single(Box::new(composite), Box::new(make_mesh()));

        let config = SolverConfiguration::new(
            TimeConfiguration::new(0.1, StepControl::Fixed { dt: 0.001 }),
            IntegratorKind::Euler,
        );

        let result = solver.solve(&scenario, &config).unwrap();
        let final_state = result.states.last().unwrap().as_scalar_field().unwrap();

        // Reference analytical solution: u(t) = exp(-3.0 * t).
        let expected = (-3.0_f64 * 0.1).exp();
        for &v in final_state.iter() {
            assert!((v - expected).abs() < 1e-2, "expected ≈{expected}, got {v}");
        }
    }

    // ── Serde round-trip (#70) ──────────────────────────────────────────────
    //
    // SplitOperator/OperatorSplittingSolver hold trait objects (Domain,
    // Box<dyn SteppableSolver>) -- not serializable, same exclusion as
    // SolverConfiguration's calculators. SplittingScheme is plain data.

    #[cfg(feature = "serde")]
    #[test]
    fn splitting_scheme_serde_roundtrip() {
        let json = serde_json::to_string(&SplittingScheme::Strang).unwrap();
        let restored: SplittingScheme = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, SplittingScheme::Strang);
    }
}
