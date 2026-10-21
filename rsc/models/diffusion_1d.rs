//! 1D diffusion model `rho*Cp*du/dt = lambda*d2u/dx2`, with its exact
//! Gaussian solution.
//!
//! Single definition shared by `examples/diffusion_1d.rs`,
//! `examples/diffusion_1d_plot.rs`, `tests/diffusion_1d_analytical.rs` and
//! `benches/diffusion_1d.rs`, each of which includes this file as a module
//! with `#[path]`. The physics and the derivation of the exact solution are
//! documented in `examples/diffusion_1d.rs`. Scenario constants (mesh size,
//! time step, end time) stay in each consumer.

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
/// under — `External` is the established convention for calculator-provided
/// quantities with no dedicated `ContextVariable` variant (see
/// `FDLaplacianCalculator::new`'s own docs).
pub(crate) fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

/// `rho*Cp*du/dt = lambda*d2u/dx2` — stored as the single ratio `alpha =
/// lambda/(rho*Cp)`, the only combination `compute_physics` needs.
pub(crate) struct Diffusion1D {
    pub(crate) alpha: f64,
    pub(crate) amplitude: f64,
    pub(crate) center: f64,
    pub(crate) sigma0: f64,
}

impl Diffusion1D {
    /// Exact solution at `(x, t)` — see `examples/diffusion_1d.rs` for the
    /// derivation. Valid for the infinite-domain problem; the finite-mesh
    /// approximation is only as good as the domain is wide relative to
    /// `sigma(t)`.
    pub(crate) fn exact(&self, x: f64, t: f64) -> f64 {
        let sigma_t = (self.sigma0.powi(2) + 2.0 * self.alpha * t).sqrt();
        let envelope = self.amplitude * self.sigma0 / sigma_t;
        let exponent = -(x - self.center).powi(2) / (2.0 * sigma_t.powi(2));
        envelope * exponent.exp()
    }
}

impl RequiresContext for Diffusion1D {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![laplacian_variable()]
    }
}

impl PhysicalModel for Diffusion1D {
    fn compute_physics(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let lap = ctx.external(laplacian_variable())?.as_scalar_field()?;
        Ok(ContextValue::ScalarField(lap.map(|v| self.alpha * v)))
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
        "diffusion_1d"
    }
}
