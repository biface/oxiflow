//! # Module `model::initial_condition`
//!
//! `PrescribedInitialState` — any [`PhysicalModel`] started from a given state
//! instead of its own initial condition.
//!
//! ## Why
//!
//! A model's [`PhysicalModel::initial_state`] is fixed by the model itself.
//! Two situations need another starting point without changing the model:
//!
//! 1. **Resuming a run** from a
//!    [`SimulationSnapshot`](crate::solver::snapshot::SimulationSnapshot): the
//!    restore contract of [DT-025](https://github.com/biface/oxiflow/issues/66)
//!    leaves it to the caller to rebuild the scenario and re-inject the saved
//!    state as the initial condition. Wrapping the model is that re-injection;
//!    [`Scenario::with_t_start`](crate::solver::scenario::Scenario::with_t_start)
//!    sets the matching start time.
//! 2. **Starting from a state computed elsewhere**: a developed flow, a
//!    perturbed equilibrium, the end state of another run.
//!
//! The wrapper changes the initial condition and nothing else: every other
//! [`RequiresContext`] and [`PhysicalModel`] method is delegated, `name()`
//! included, so the result of a run keeps the wrapped model's name.
//!
//! ## What a resume reproduces
//!
//! A run resumed at step N from a snapshot and continued for M steps matches
//! an uninterrupted N + M run under two conditions:
//!
//! - **One-step integrators only.** A snapshot carries no history, so a
//!   multistep integrator (BDF2) restarts on its bootstrap step and the two
//!   runs part from there.
//! - **Times agree to rounding, not bit for bit.** The fixed-step loop
//!   computes each time as `t_start + k·dt`. Resumed from `t_start = t_N`, it
//!   computes `t_N + k·dt` where the uninterrupted run computes
//!   `(N + k)·dt`; the two can differ in the last bits. States are identical
//!   for a model that does not read the time; a model that does sees those
//!   last-bit differences.
//!
//! ## Shape
//!
//! [`initial_state`](PhysicalModel::initial_state) cannot report an error, so
//! the prescribed state is checked against the mesh by
//! [`Scenario::validate`](crate::solver::scenario::Scenario::validate), which
//! [`Solver::solve`](crate::solver::Solver::solve) and the orchestrator call
//! before the first step. A caller that drives
//! [`SteppableSolver::step`](crate::solver::methods::SteppableSolver::step)
//! in its own loop calls it itself.
//!
//! # Examples
//!
//! Resuming a run from a snapshot taken at `t = 0.5`:
//!
//! ```rust
//! use nalgebra::DVector;
//! use oxiflow::context::{ComputeContext, ContextValue, ContextVariable, OxiflowError};
//! use oxiflow::mesh::{Mesh, UniformGrid1D};
//! use oxiflow::model::{PhysicalModel, PrescribedInitialState, RequiresContext};
//! use oxiflow::solver::config::{IntegratorKind, StepControl, TimeConfiguration};
//! use oxiflow::solver::methods::ForwardEulerSolver;
//! use oxiflow::solver::scenario::Scenario;
//! use oxiflow::solver::snapshot::SimulationSnapshot;
//! use oxiflow::solver::{Solver, SolverConfiguration};
//!
//! struct Decay;
//! impl RequiresContext for Decay {
//!     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
//! }
//! impl PhysicalModel for Decay {
//!     fn compute_physics(&self, s: &ContextValue, _: &ComputeContext)
//!         -> Result<ContextValue, OxiflowError> {
//!         Ok(ContextValue::ScalarField(s.as_scalar_field()?.map(|v| -v)))
//!     }
//!     fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
//!         ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
//!     }
//!     fn name(&self) -> &str { "decay" }
//! }
//!
//! let time = TimeConfiguration::new(0.5, StepControl::Fixed { dt: 0.125 });
//! let first = Scenario::single(Box::new(Decay), Box::new(UniformGrid1D::new(4, 0.0, 1.0)?));
//! let run = ForwardEulerSolver.solve(&first, &SolverConfiguration::new(time.clone(), IntegratorKind::Euler))?;
//!
//! // Explicit checkpoint at the end of the first run.
//! let snapshot = SimulationSnapshot {
//!     time_config: time,
//!     integrator: IntegratorKind::Euler,
//!     state: run.states.last().unwrap().clone(),
//!     t: run.t_final().unwrap(),
//!     n_steps: run.n_steps,
//!     error: None,
//!     partial: None,
//! };
//!
//! // Rebuild the scenario, re-inject the state, restart from its time.
//! let model = PrescribedInitialState::new(Box::new(Decay), snapshot.state.clone());
//! let resumed = Scenario::single(Box::new(model), Box::new(UniformGrid1D::new(4, 0.0, 1.0)?))
//!     .with_t_start(snapshot.t);
//! let config = SolverConfiguration::new(
//!     TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.125 }),
//!     IntegratorKind::Euler,
//! );
//! let rest = ForwardEulerSolver.solve(&resumed, &config)?;
//! assert_eq!(rest.states[0], snapshot.state);
//! assert_eq!(rest.t_final(), Some(1.0));
//! # Ok::<(), OxiflowError>(())
//! ```

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::{PhysicalModel, RequiresContext};

/// A [`PhysicalModel`] whose initial condition is replaced by a given state.
///
/// See the [module documentation](self) for the restore contract it serves
/// and what a resumed run reproduces.
pub struct PrescribedInitialState {
    model: Box<dyn PhysicalModel>,
    state: ContextValue,
}

impl PrescribedInitialState {
    /// Wraps `model` so that its initial condition is `state`.
    ///
    /// `state` is not checked here: the mesh is not known yet.
    /// [`Scenario::validate`](crate::solver::scenario::Scenario::validate)
    /// checks it against the domain's mesh (see the
    /// [module documentation](self#shape) for who calls it).
    pub fn new(model: Box<dyn PhysicalModel>, state: ContextValue) -> Self {
        Self { model, state }
    }

    /// The wrapped model.
    pub fn model(&self) -> &dyn PhysicalModel {
        self.model.as_ref()
    }

    /// The prescribed initial state.
    pub fn state(&self) -> &ContextValue {
        &self.state
    }
}

impl RequiresContext for PrescribedInitialState {
    fn required_variables(&self) -> Vec<ContextVariable> {
        self.model.required_variables()
    }

    fn optional_variables(&self) -> Vec<ContextVariable> {
        self.model.optional_variables()
    }

    fn depends_on(&self) -> Vec<ContextVariable> {
        self.model.depends_on()
    }

    fn priority(&self) -> u32 {
        self.model.priority()
    }
}

impl PhysicalModel for PrescribedInitialState {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        self.model.compute_physics(state, ctx)
    }

    /// Returns the prescribed state, whatever the mesh: its size is checked
    /// by `Scenario::validate`.
    fn initial_state(&self, _mesh: &dyn Mesh) -> ContextValue {
        self.state.clone()
    }

    fn name(&self) -> &str {
        self.model.name()
    }

    fn description(&self) -> Option<&str> {
        self.model.description()
    }

    fn max_wave_speed(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<Option<f64>, OxiflowError> {
        self.model.max_wave_speed(state, ctx)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::structured::UniformGrid1D;
    use crate::model::FullModel;
    use nalgebra::{DMatrix, DVector};

    fn prescribed() -> PrescribedInitialState {
        PrescribedInitialState::new(
            Box::new(FullModel),
            ContextValue::ScalarField(DVector::from_vec(vec![3.0, 4.0, 5.0])),
        )
    }

    #[test]
    fn context_requirements_are_those_of_the_wrapped_model() {
        let model = prescribed();
        assert_eq!(model.required_variables(), FullModel.required_variables());
        assert_eq!(model.optional_variables(), FullModel.optional_variables());
        assert_eq!(model.depends_on(), FullModel.depends_on());
        assert_eq!(model.priority(), FullModel.priority());
    }

    #[test]
    fn name_and_description_are_those_of_the_wrapped_model() {
        let model = prescribed();
        assert_eq!(model.name(), FullModel.name());
        assert_eq!(model.description(), FullModel.description());
    }

    #[test]
    fn wave_speed_is_that_of_the_wrapped_model() {
        let ctx = ComputeContext::new(0.0, 0.01);
        let state = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let declared = PrescribedInitialState::new(
            Box::new(crate::model::DeclaredWaveSpeed { speed: 3.0 }),
            state.clone(),
        );
        assert_eq!(declared.max_wave_speed(&state, &ctx).unwrap(), Some(3.0));
        assert_eq!(prescribed().max_wave_speed(&state, &ctx).unwrap(), None);
    }

    #[test]
    fn physics_is_that_of_the_wrapped_model() {
        let ctx = ComputeContext::new(0.0, 0.01);
        let state = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(
            prescribed().compute_physics(&state, &ctx).unwrap(),
            FullModel.compute_physics(&state, &ctx).unwrap()
        );
    }

    #[test]
    fn initial_state_is_the_prescribed_state_not_the_model_one() {
        let mesh = UniformGrid1D::new(3, 0.0, 1.0).unwrap();
        let model = prescribed();
        assert_eq!(model.initial_state(&mesh), *model.state());
        assert_ne!(model.initial_state(&mesh), FullModel.initial_state(&mesh));
    }

    #[test]
    fn a_vector_field_state_is_returned_unchanged() {
        let mesh = UniformGrid1D::new(2, 0.0, 1.0).unwrap();
        let state = ContextValue::VectorField(DMatrix::from_row_slice(2, 2, &[1., 2., 3., 4.]));
        let model = PrescribedInitialState::new(Box::new(FullModel), state.clone());
        assert_eq!(model.initial_state(&mesh), state);
        assert_eq!(model.model().name(), FullModel.name());
    }
}
