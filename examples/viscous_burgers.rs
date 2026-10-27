//! # Example: viscous Burgers benchmark model (#137, feeds #53)
//!
//! `du/dt + u*du/dx = nu*d2u/dx2` on `UniformGrid1D`, via `FDGradientCalculator`
//! (centered, for the nonlinear advective term) and `FDLaplacianCalculator`
//! (for the diffusive term) -- one nonlinear term heavier per node than
//! diffusion_1d's bare stencil, chosen for the middle weight class of the
//! #53 benchmark family.
//!
//! Run with: `cargo run --example viscous_burgers`
//!
//! The model (`ViscousBurgers` and its exact solution) is defined once in
//! `rsc/models/viscous_burgers.rs`, included below as the
//! `viscous_burgers_model` module and shared with the plot example, the
//! analytical test and the benchmark.
//!
//! ## Exact solution: traveling viscous shock
//!
//! The Cole-Hopf transform linearizes Burgers' equation into the heat
//! equation, giving several known closed-form solutions (Hopf, 1950; Cole,
//! 1951). Rather than the periodic Fourier-Bessel series solution commonly
//! used for numerical-method benchmarking (Basdevant et al., *Spectral and
//! finite difference solutions of the Burgers equation*, 1986), this model
//! uses the simpler traveling viscous shock connecting two asymptotic
//! states `u_L` (as `x -> -inf`) and `u_R` (as `x -> +inf`):
//!
//! ```text
//! u(x,t) = (u_L+u_R)/2 - (u_L-u_R)/2 * tanh[ (u_L-u_R)/(4*nu) * (x - x0 - (u_L+u_R)/2*t) ]
//! ```
//!
//! A smooth front of half-width ~`4*nu/(u_L-u_R)`, translating at speed
//! `(u_L+u_R)/2` -- exact for the infinite-domain problem, same finite-mesh
//! caveat as `examples/diffusion_1d.rs`: the domain must stay wide enough
//! relative to the front's width that the one-sided edge stencils
//! (`CenteredGradient`/`CenteredLaplacian`) never see a non-negligible
//! signal. Verified independently (outside oxiflow, a standalone FD
//! replica) before use here: the scheme converges toward this formula at
//! ~2nd order as the mesh is refined, confirming it solves the *nonlinear*
//! PDE, not just a linearized approximation of it.
//!
//! See `tests/viscous_burgers_analytical.rs` for the accuracy and
//! convergence-order assertions.

use std::sync::Arc;

use oxiflow::{
    context::calculators::{FDGradientCalculator, FDLaplacianCalculator, FDScheme},
    mesh::{Mesh, UniformGrid1D},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};

#[path = "../rsc/models/viscous_burgers.rs"]
mod viscous_burgers_model;

use viscous_burgers_model::{laplacian_variable, ViscousBurgers};

fn main() {
    // Front half-width ~4*NU/(U_L-U_R) = 0.2; domain half-width 3.0 keeps
    // both edges >=8 half-widths from the front's path (front travels from
    // x0=-2.0 to x0+SPEED*T_END = -1.0 over the run) -- comfortably inside
    // the "10x" margin used for diffusion_1d, though the exact multiple
    // differs since the front also translates, not just spreads in place.
    const N_NODES: usize = 401;
    const NU: f64 = 0.05;
    const U_L: f64 = 1.0;
    const U_R: f64 = 0.0;
    const X0: f64 = -2.0;
    const T_END: f64 = 2.0;
    const DOMAIN_HALF_WIDTH: f64 = 3.0;

    let model = ViscousBurgers {
        nu: NU,
        u_l: U_L,
        u_r: U_R,
        x0: X0,
    };
    let mesh = UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap();
    let dx = 2.0 * DOMAIN_HALF_WIDTH / (N_NODES as f64 - 1.0);
    let gradient_mesh: Arc<dyn Mesh> =
        Arc::new(UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap());
    let laplacian_mesh: Arc<dyn Mesh> =
        Arc::new(UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap());

    // dt scaled with dx^2/nu (diffusive stability) -- verified during
    // design to give clean ~2nd-order convergence, unlike a fixed dt
    // (see diffusion_1d's own dt-sweep note for why fixed dt hides
    // spatial convergence behind temporal error at fine resolutions).
    let dt = 0.4 * dx * dx / NU;

    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt }),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDGradientCalculator::new(
        gradient_mesh,
        0,
        None,
        FDScheme::Central,
    )))
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        laplacian_mesh,
        laplacian_variable(),
    )));

    let solver = ForwardEulerSolver;
    let result = solver.solve(&scenario, &config).unwrap();
    let final_state = result.states.last().unwrap().as_scalar_field().unwrap();

    let mesh_ref = UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap();
    let exact_model = ViscousBurgers {
        nu: NU,
        u_l: U_L,
        u_r: U_R,
        x0: X0,
    };
    let max_error = (0..N_NODES)
        .map(|i| {
            let x = mesh_ref.coordinates(i)[0];
            (final_state[i] - exact_model.exact(x, T_END)).abs()
        })
        .fold(0.0, f64::max);

    println!("viscous_burgers: N={N_NODES}, t_end={T_END}, max |error| = {max_error:e}");
}
