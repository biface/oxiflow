//! Viscous Burgers model `du/dt + u*du/dx = nu*d2u/dx2`, with its exact
//! traveling-shock solution.
//!
//! Single definition shared by `examples/viscous_burgers.rs`,
//! `examples/viscous_burgers_plot.rs`, `tests/viscous_burgers_analytical.rs`
//! and `benches/viscous_burgers.rs`, each of which includes this file as a
//! module with `#[path]`. The physics, the Cole-Hopf exact solution and its
//! finite-domain caveat are documented in `examples/viscous_burgers.rs`.
//! Scenario constants (mesh size, domain, end time) stay in each consumer.

use nalgebra::DVector;
use oxiflow::{
    context::{
        compute::ComputeContext, error::OxiflowError, value::ContextValue,
        variable::ContextVariable,
    },
    mesh::Mesh,
    model::traits::{PhysicalModel, RequiresContext},
};

/// The `ContextVariable` slot `FDLaplacianCalculator` publishes its result
/// under (`External`, as in `rsc/models/diffusion_1d.rs`).
pub(crate) fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

/// `du/dt + u*du/dx = nu*d2u/dx2`.
pub(crate) struct ViscousBurgers {
    pub(crate) nu: f64,
    pub(crate) u_l: f64,
    pub(crate) u_r: f64,
    pub(crate) x0: f64,
}

impl ViscousBurgers {
    /// Exact traveling-shock solution at `(x, t)` -- see
    /// `examples/viscous_burgers.rs` for the derivation and its
    /// finite-domain caveat.
    pub(crate) fn exact(&self, x: f64, t: f64) -> f64 {
        let speed = (self.u_l + self.u_r) / 2.0;
        let half_jump = (self.u_l - self.u_r) / 2.0;
        let arg = (self.u_l - self.u_r) / (4.0 * self.nu) * (x - self.x0 - speed * t);
        speed - half_jump * arg.tanh()
    }
}

impl RequiresContext for ViscousBurgers {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            laplacian_variable(),
        ]
    }
}

impl PhysicalModel for ViscousBurgers {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        let du_dx = ctx.gradient(0)?;
        let lap = ctx.external(laplacian_variable())?.as_scalar_field()?;

        // du/dt = -u*du/dx + nu*d2u/dx2 -- no boundary special-casing here:
        // FDGradientCalculator/FDLaplacianCalculator already handle their
        // own edge stencils internally, same as diffusion_1d.
        let du_dt = -u.component_mul(du_dx) + lap.map(|v| self.nu * v);
        Ok(ContextValue::ScalarField(du_dt))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        let u0 = DVector::from_iterator(
            mesh.n_dof(),
            (0..mesh.n_dof()).map(|i| {
                let x = mesh.coordinates(i)[0];
                self.exact(x, 0.0)
            }),
        );
        ContextValue::ScalarField(u0)
    }

    fn name(&self) -> &str {
        "viscous_burgers"
    }
}
