//! # Module `solver::methods::imex`
//!
//! Opérateur splitting temporel — `OperatorSplittingSolver` (DD-037, #45).
//!
//! ## Couplage temporel, pas spatial
//!
//! Ce module compose **n ≥ 2** [`PhysicalModel`](crate::model::PhysicalModel)
//! évalués en séquence sur le **même état**, le **même maillage** — par
//! opposition au couplage spatial d'INV-3 (`CouplingOperator`, cross-domain,
//! asymétrique source → cible à une `Interface`). Voir DD-037 pour la
//! distinction complète entre les deux familles.
//!
//! La forme canonique d'oxiflow nomme déjà deux termes séparés :
//!
//! $$\frac{\partial u}{\partial t} + \nabla \cdot F(u, \nabla u) = S(u, \mathbf{x}, t)$$
//!
//! Chaque [`SplitOperator`] porte **son propre [`Domain`]** (même maillage,
//! mêmes BC le cas échéant, modèle différent) plutôt qu'un simple modèle :
//! [`SteppableSolver::step`] lit toujours `domain.model`, donc faire porter
//! chaque contribution par son propre `Domain` suffit à l'isoler
//! correctement, sans toucher à `SteppableSolver` ni à aucun solveur
//! existant.
//!
//! ## Portée v1 (#45)
//!
//! Seuls des sous-solveurs **à un pas** (`history_depth() == 0`) sont
//! supportés — validé au constructeur. [`SplittingScheme::Strang`] est la
//! seule variante implémentée et testée ; [`SplittingScheme::LieTrotter`]
//! est réservée (DD-037) et refusée au constructeur tant qu'aucun cas
//! concret ne l'exige.
//!
//! ## Le `Scenario`/`Domain` extérieur
//!
//! [`Solver::solve`] impose un `Scenario` — donc un `Domain` extérieur —
//! même si `OperatorSplittingSolver` n'en a pas besoin pour calculer (il a
//! tout dans ses propres opérateurs). DD-037 tranche pour
//! [`crate::model::CompositeModel`] : le `Domain` extérieur porte la somme
//! des contributions, ce qui rend `scenario.context_requirements()` et
//! `domain.model.initial_state()` corrects sans logique dédiée ici — et
//! sert aussi de référence monolithique testable par un solveur ordinaire
//! (critère d'acceptation 1 de #45).

use std::collections::HashMap;

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::StepControl;
use crate::solver::methods::{check_finite, SteppableSolver};
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// Une contribution nommée à `∂u/∂t`, intégrée par son propre sous-solveur.
///
/// Porte un [`Domain`] complet (pas seulement un modèle) — voir la doc de
/// module pour pourquoi.
pub struct SplitOperator {
    /// Maillage, BC, et modèle propres à cette contribution.
    pub domain: Domain,
    /// Sous-solveur intégrant cette contribution. Doit avoir
    /// `history_depth() == 0` (portée v1, DD-037).
    pub solver: Box<dyn SteppableSolver>,
}

/// Schéma de composition des opérateurs sur un pas `dt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SplittingScheme {
    /// Pas complet de chaque opérateur, en séquence — ordre 1.
    ///
    /// Réservée (DD-037) : même structure que `Strang`, mais non implémentée
    /// ni testée ici — `OperatorSplittingSolver::new` la refuse tant
    /// qu'aucun cas concret ne l'exige.
    LieTrotter,
    /// Demi-pas sur chaque opérateur sauf le dernier (pas complet), puis
    /// repasse en ordre inverse — composition symétrique/palindromique,
    /// ordre 2. Pour n=2, c'est exactement le schéma de #45 : demi-pas
    /// explicite → pas implicite complet → demi-pas explicite.
    Strang,
}

/// Composite générique : n ≥ 2 opérateurs partageant le même état,
/// composés selon un [`SplittingScheme`] (DD-037, #45).
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
    /// Construit un composite à partir d'au moins deux opérateurs.
    ///
    /// # Errors
    ///
    /// `OxiflowError::PreconditionFailed` si :
    /// - moins de deux opérateurs sont fournis ;
    /// - un sous-solveur a `history_depth() != 0` (portée v1, DD-037) ;
    /// - `scheme` est `SplittingScheme::LieTrotter` (réservée, non
    ///   implémentée — DD-037).
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

    /// Constructeur ergonomique pour le cas n=2 demandé par #45 : demi-pas
    /// explicite → pas implicite complet → demi-pas explicite.
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

    /// Avance `state` d'un pas extérieur `dt`, selon `self.scheme`.
    fn apply_step(
        &self,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        match self.scheme {
            // Non atteignable : refusée par `new` — gardé pour exhaustivité
            // si `SplittingScheme` gagne des variantes plus tard (DD-037).
            SplittingScheme::LieTrotter => unreachable!(
                "SplittingScheme::LieTrotter is rejected by OperatorSplittingSolver::new"
            ),
            SplittingScheme::Strang => self.apply_strang(chain, state, t, dt),
        }
    }

    /// Composition symétrique/palindromique, généralisée à n ≥ 2 (DD-037).
    ///
    /// Φ₁(dt/2) ∘ Φ₂(dt/2) ∘ … ∘ Φₙ₋₁(dt/2) ∘ Φₙ(dt) ∘ Φₙ₋₁(dt/2) ∘ … ∘ Φ₁(dt/2)
    ///
    /// Pour n=2 : Φ₁ demi-pas (explicite) → Φ₂ pas complet (implicite) →
    /// Φ₁ demi-pas — exactement le schéma de #45.
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

        // Passe avant : demi-pas sur operators[0..n-1].
        for op in &self.operators[..n - 1] {
            current = op
                .solver
                .step(&op.domain, chain, &mut current, &[], t_local, half)?;
            t_local += half;
        }

        // Pas complet sur le dernier opérateur.
        let last = &self.operators[n - 1];
        current = last
            .solver
            .step(&last.domain, chain, &mut current, &[], t_local, dt)?;
        t_local += dt;

        // Passe retour : demi-pas sur operators[0..n-1], ordre inverse.
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
    /// Boucle à pas fixe — ne réutilise pas
    /// [`SteppableSolver::solve_fixed_step`] (DD-035) : ce composite n'est
    /// pas un `SteppableSolver` (même posture que DD-036 pour DoPri45 —
    /// éviter de rouvrir l'orchestration multi-domaine pour un besoin non
    /// posé par #45). La boucle ci-dessous reprend volontairement la même
    /// forme que `solve_fixed_step` pour rester cohérente avec les autres
    /// solveurs à pas fixe.
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

    /// `du/dt = -rate * u` — décroissance pure, pas de dépendance au temps.
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
        // Deux décroissances séparées (rate 1.0 + rate 2.0) splittées doivent
        // approcher la décroissance combinée (rate 3.0) — critère
        // d'acceptation 1 de #45.
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

        // Solution analytique de référence : u(t) = exp(-3.0 * t).
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
