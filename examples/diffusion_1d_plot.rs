//! # Example: diffusion_1d reference plot (#145)
//!
//! `u(x)` profiles at several times, overlaid with the analytical
//! Gaussian solution -- the standard textbook figure for a 1D diffusion
//! equation (Carslaw & Jaeger, *Conduction of Heat in Solids*, 1959;
//! Crank, *The Mathematics of Diffusion*, 1975), same convention used in
//! `examples/diffusion_1d.rs`'s own module doc.
//!
//! Model code duplicated from `examples/diffusion_1d.rs` (same
//! convention already established for `tests/`/`benches/` -- see that
//! file's own module doc for the physics).
//!
//! Analytical solution rendered as sparse circle markers rather than a
//! dashed line: `plotters` has no simple, reliable dashed-line style on
//! its base line series, and marker-vs-line is an equally standard
//! literature convention for numerical-vs-analytical overlays.
//!
//! Run with: `cargo run --example diffusion_1d_plot` -- writes
//! `diffusion_1d.svg` to the working directory.

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
use plotters::prelude::*;

fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

struct Diffusion1D {
    alpha: f64,
    amplitude: f64,
    center: f64,
    sigma0: f64,
}

impl Diffusion1D {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_NODES: usize = 401;
    const ALPHA: f64 = 0.01;
    const AMPLITUDE: f64 = 1.0;
    const CENTER: f64 = 0.0;
    const SIGMA0: f64 = 0.05;
    const T_END: f64 = 0.2;
    const DT: f64 = 1e-5;
    // 5 snapshots + the initial state -- 20,000 total steps at DT=1e-5.
    const N_SNAPSHOTS: usize = 5;
    const SAVE_EVERY: usize = (T_END / DT) as usize / N_SNAPSHOTS;

    let model = Diffusion1D {
        alpha: ALPHA,
        amplitude: AMPLITUDE,
        center: CENTER,
        sigma0: SIGMA0,
    };
    let mesh = UniformGrid1D::new(N_NODES, -1.0, 1.0)?;
    let mesh_arc: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(N_NODES, -1.0, 1.0)?);
    let plot_mesh = UniformGrid1D::new(N_NODES, -1.0, 1.0)?;
    let x_coords: Vec<f64> = (0..N_NODES).map(|i| plot_mesh.coordinates(i)[0]).collect();

    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }).saving_every(SAVE_EVERY),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        mesh_arc,
        laplacian_variable(),
    )));

    let solver = ForwardEulerSolver;
    let result = solver.solve(&scenario, &config)?;

    let exact_model = Diffusion1D {
        alpha: ALPHA,
        amplitude: AMPLITUDE,
        center: CENTER,
        sigma0: SIGMA0,
    };

    // Plot bounds: y in [0, amplitude] with 10% headroom, x the full domain.
    let root = SVGBackend::new("diffusion_1d.svg", (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "diffusion_1d: u(x,t) vs. analytical Gaussian solution",
            ("sans-serif", 24),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(-1.0f64..1.0f64, 0.0f64..AMPLITUDE * 1.1)?;
    chart.configure_mesh().x_desc("x").y_desc("u").draw()?;

    let palette = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];
    for (idx, (t, state)) in result.times.iter().zip(&result.states).enumerate() {
        let color = palette[idx % palette.len()];
        let field = state.as_scalar_field()?;

        chart
            .draw_series(LineSeries::new(
                x_coords.iter().zip(field.iter()).map(|(&x, &u)| (x, u)),
                color,
            ))?
            .label(format!("t={t:.3} (simulated)"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));

        // Analytical reference: sparse circle markers, every 20th node.
        chart.draw_series(
            x_coords
                .iter()
                .step_by(20)
                .map(|&x| Circle::new((x, exact_model.exact(x, *t)), 3, color.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;

    println!(
        "diffusion_1d_plot: wrote diffusion_1d.svg ({} snapshots)",
        result.states.len()
    );
    Ok(())
}
