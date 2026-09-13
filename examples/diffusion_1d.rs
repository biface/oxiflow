//! # Example: 1D diffusion benchmark model (#136, feeds #53)
//!
//! Demonstrates the lightest-weight benchmark subject in the #53 family:
//! `rho*Cp*du/dt = lambda*d2u/dx2` on `UniformGrid1D`, via
//! `FDLaplacianCalculator`. Per-node cost is a single FD stencil, no
//! nonlinear term — the cheapest of the four benchmark models (diffusion,
//! Burgers, Langmuir, lahar), chosen to anchor the light end of the
//! parallel-threshold calibration range.
//!
//! Run with: `cargo run --example diffusion_1d`
//!
//! ## Physics and exact solution
//!
//! `alpha = lambda / (rho*Cp)` is the thermal diffusivity — the model
//! stores `alpha` directly rather than the three separate physical
//! constants, since `compute_physics` only ever uses their ratio.
//!
//! Initial condition: a Gaussian pulse `u(x,0) = A * exp(-(x-x0)^2 /
//! (2*sigma0^2))`. For the *linear* 1D heat equation on an infinite
//! domain, a Gaussian initial condition stays Gaussian for all `t`, with
//! variance growing linearly in `t` (Carslaw & Jaeger, *Conduction of
//! Heat in Solids*, 1959; Crank, *The Mathematics of Diffusion*, 1975):
//!
//! ```text
//! sigma(t)^2 = sigma0^2 + 2*alpha*t
//! u(x,t) = A * sigma0/sigma(t) * exp(-(x-x0)^2 / (2*sigma(t)^2))
//! ```
//!
//! This is exact only for an *infinite* domain. On the finite
//! `UniformGrid1D` used here, the domain half-width is chosen at least
//! 10x `sigma(t_end)` so the pulse's tails are negligible at both edges
//! for the whole run — the one-sided edge stencil `CenteredLaplacian`
//! falls back to at the two boundary nodes (same "reuse the nearest
//! interior formula" convention as `operators::fv`'s `Truncation`
//! boundary) never sees a non-negligible signal, so it does not
//! contaminate the comparison against the infinite-domain solution.
//!
//! See `tests/diffusion_1d_analytical.rs` for the accuracy and
//! convergence-order assertions this example's numbers are not, by
//! themselves, a regression gate for.

use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    context::{
        calculators::FDLaplacianCalculator, compute::ComputeContext, error::OxiflowError,
        value::ContextValue, variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};

/// The `ContextVariable` slot `FDLaplacianCalculator` publishes its result
/// under — `External` is the established convention for calculator-provided
/// quantities with no dedicated `ContextVariable` variant (see
/// `FDLaplacianCalculator::new`'s own docs).
fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

/// `rho*Cp*du/dt = lambda*d2u/dx2` — stored as the single ratio `alpha =
/// lambda/(rho*Cp)`, the only combination `compute_physics` needs.
struct Diffusion1D {
    alpha: f64,
    amplitude: f64,
    center: f64,
    sigma0: f64,
}

impl Diffusion1D {
    /// Exact solution at `(x, t)` — see the module documentation for the
    /// derivation. Valid for the infinite-domain problem; the finite-mesh
    /// approximation here is only as good as the domain is wide relative
    /// to `sigma(t)`.
    fn exact(&self, x: f64, t: f64) -> f64 {
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

fn main() {
    // Domain half-width 1.0 against sigma(T_END) ~ 0.08 (see the module
    // documentation's margin calculation) -- comfortably within the "10x"
    // criterion.
    const N_NODES: usize = 401;
    const ALPHA: f64 = 0.01;
    const AMPLITUDE: f64 = 1.0;
    const CENTER: f64 = 0.0;
    const SIGMA0: f64 = 0.05;
    const T_END: f64 = 0.2;
    // Small enough that Euler's O(dt) temporal error stays well below the
    // stencil's O(dx^2) spatial error at this resolution -- verified by a
    // dt sweep during design: at dt=1e-3 the fixed-dt error floor masks
    // spatial convergence entirely (order collapses to ~0.2 across a
    // mesh doubling); dt=1e-5 gives a clean, consistent order ~2.0-2.1
    // across two doublings. See tests/diffusion_1d_analytical.rs for the
    // convergence-order regression check this value feeds.
    const DT: f64 = 1e-5;

    let model = Diffusion1D {
        alpha: ALPHA,
        amplitude: AMPLITUDE,
        center: CENTER,
        sigma0: SIGMA0,
    };
    let mesh = UniformGrid1D::new(N_NODES, -1.0, 1.0).unwrap();
    let mesh_arc: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(N_NODES, -1.0, 1.0).unwrap());

    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        mesh_arc,
        laplacian_variable(),
    )));

    let solver = ForwardEulerSolver;
    let result = solver.solve(&scenario, &config).unwrap();
    let final_state = result.states.last().unwrap().as_scalar_field().unwrap();

    // Report against the analytical solution -- diagnostic only, see
    // tests/diffusion_1d_analytical.rs for the actual regression gate.
    let mesh_ref = UniformGrid1D::new(N_NODES, -1.0, 1.0).unwrap();
    let exact_model = Diffusion1D {
        alpha: ALPHA,
        amplitude: AMPLITUDE,
        center: CENTER,
        sigma0: SIGMA0,
    };
    let max_error = (0..N_NODES)
        .map(|i| {
            let x = mesh_ref.coordinates(i)[0];
            (final_state[i] - exact_model.exact(x, T_END)).abs()
        })
        .fold(0.0, f64::max);

    println!("diffusion_1d: N={N_NODES}, t_end={T_END}, max |error| = {max_error:e}");
}
