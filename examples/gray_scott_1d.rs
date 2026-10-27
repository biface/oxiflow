//! # Reference example: Gray–Scott in 1D, Turing instability and self-replicating pulses (#176)
//!
//! Two reacting and diffusing species on a ring. The example runs two cases
//! of the same model:
//!
//! - **Turing instability.** A homogeneous state that is stable on its own
//!   becomes unstable to a band of spatial modes when the two species
//!   diffuse at different rates. Single modes laid on that state grow or
//!   decay at rates given exactly by the linearised equations; the example
//!   checks the measured rates.
//! - **Self-replicating pulses.** A local perturbation of the trivial state
//!   grows into a pulse of the autocatalyst, which splits in two; the outer
//!   pulses split again, and the replication goes on until the ring is
//!   filled. The example checks the pulse count at three times and the
//!   mirror symmetry of the state.
//!
//! Run with: `cargo run --example gray_scott_1d`. The example exits with an
//! error if a check exceeds its tolerance.
//!
//! ## Physics
//!
//! Gray–Scott model of cubic autocatalysis, U + 2V -> 3V and V -> P, fed
//! with U and drained (Gray and Scott, 1983). It is a model system with no
//! material behind it, written in dimensionless form with the scales of
//! Pearson (1993): diffusivities Du = 2e-5 and Dv = 1e-5, a feed rate F
//! and a removal rate k.
//!
//! ```text
//! du/dt = Du * d2u/dx2 - u v² + F (1 - u)
//! dv/dt = Dv * d2v/dx2 + u v² - (F + k) v,    x on a ring of period 2.5
//! ```
//!
//! The model is defined in this file: no other target shares it. Its state
//! is a two-column `VectorField` (u, v); it reads the Laplacian of each
//! column from an `FDLaplacianCalculator` (`with_component`), second order
//! in space.
//!
//! ## Periodic closure
//!
//! The ring is a grid of n nodes with spacing dx = 2.5 / n, node n - 1
//! being followed by node 0. `FDLaplacianCalculator` takes a one-sided
//! stencil at the two end nodes, which imposes no condition; the model
//! replaces its value there with the centred stencil across the closure,
//! (u[n-1] - 2 u[0] + u[1]) / dx² and (u[n-2] - 2 u[n-1] + u[0]) / dx².
//! The ring is a property of this model, not a boundary condition of the
//! crate: oxiflow has no periodic condition for finite differences, and
//! its boundary conditions apply to scalar states only.
//!
//! ## Case 1: Turing instability
//!
//! (F, k) = (0.066, 0.06245), on the upper branch of homogeneous states:
//! v* is the larger root of (F + k) v² - F v + F (F + k) = 0 and
//! u* = (F + k) / v*, so u* = 0.49699, v* = 0.25846. The Jacobian of the
//! reaction terms there,
//!
//! ```text
//! J = | -v*² - F    -2 u* v*          |
//!     |  v*²         2 u* v* - (F + k) |
//! ```
//!
//! has eigenvalues of real part -2.2e-3: the state is stable to uniform
//! perturbations. On the grid, the mode cos(2 pi m i / n) is an
//! eigenfunction of the discrete Laplacian, with eigenvalue -q², where
//! q² = (4 / dx²) sin²(pi m / n). Laid on the homogeneous state along an
//! eigenvector of J - q² diag(Du, Dv), the mode grows as exp(sigma t),
//! sigma being the matching eigenvalue: positive for m from 4 to 31 (the
//! Turing band), largest at m = 16.
//!
//! The example runs four modes, each on its own: 5 and 31 near the edges
//! of the band, 16 at its peak, and 40, a decaying mode. Modes 0 to 3 are
//! left out: their eigenvalues are complex (damped oscillations). Each run
//! starts from the homogeneous state plus an amplitude 1e-7 of the mode,
//! on 256 nodes with RK4 and dt = 1, up to t = 300. The measured rate is
//! ln(A(300) / A(0)) / 300, A being the projection of v - v* on the mode.
//! The example also prints the rate of the continuous equations (q² of
//! the continuous Laplacian, (2 pi m / 2.5)²), not checked: the difference
//! shows the spatial error of the grid, which the discrete q² removes.
//!
//! ## Case 2: self-replicating pulses
//!
//! (F, k) = (0.04, 0.06). The trivial state u = 1, v = 0 is stable; the
//! initial state is that state except on the 21 nodes within 0.05 of the
//! middle of the ring, where u = 1/2 and v = 1/4 (the perturbation of
//! Pearson, 1993). In this regime, the perturbation grows into a pulse of
//! v that splits in two; each cycle, the two outermost pulses split again,
//! until the ring is filled with a periodic array of pulses (Reynolds,
//! Pearson and Ponce-Dawson, 1994). 512 nodes, RK4, dt = 0.5, up to
//! t = 2500, a state saved every 100 time units.
//!
//! A pulse is a run of consecutive nodes where v exceeds 0.2 (peak values
//! stay between 0.31 and 0.35). The counts at t = 600, 1500 and 2500 fall
//! in the middle of the plateaus between splittings: 2, 4 and 6 on 256,
//! 512 and 1024 nodes alike, with the threshold 0.1 or 0.2, when the
//! example was written. The splitting times themselves move with the grid
//! by a few tens of time units.
//!
//! The initial state and the ring are symmetric under the reflection
//! i -> n - i about the middle node, so the state stays symmetric; the
//! example checks the largest asymmetry of u and v over the saved states.
//!
//! ## Validation
//!
//! | Check | Tolerance | Measured when written |
//! |---|---|---|
//! | Growth rate of modes 5, 16, 31 and 40, relative error | 1e-7 | 5.0e-9, 4.3e-9, 1.4e-10, 2.8e-8 |
//! | Pulse count at t = 600, 1500, 2500 | exact: 2, 4, 6 | 2, 4, 6 |
//! | Mirror asymmetry of u and v | 1e-12 | 4e-15 |
//!
//! The growth-rate errors do not depend on the time step (dt = 0.5 gives
//! the same values). They come from the nonlinear terms, which grow with
//! the square of the mode amplitude (an amplitude of 1e-6 gives 5.6e-7 for
//! mode 16), and from rounding for the decaying mode 40, whose amplitude
//! ends near 8e-11.
//!
//! References: P. Gray and S. K. Scott, *Chem. Eng. Sci.* 38, 29 (1983);
//! J. E. Pearson, *Science* 261, 189 (1993); W. N. Reynolds, J. E.
//! Pearson and S. Ponce-Dawson, *Phys. Rev. Lett.* 72, 2797 (1994).

use std::error::Error;
use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::DMatrix;
use oxiflow::{
    context::{
        calculators::FDLaplacianCalculator, compute::ComputeContext, error::OxiflowError,
        value::ContextValue, variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::RK4Solver,
        scenario::Scenario,
        SimulationResult, Solver, SolverConfiguration,
    },
};

// ── Model ─────────────────────────────────────────────────────────────────

/// Ring period.
const PERIOD: f64 = 2.5;
/// Diffusivity of u (substrate).
const DIFFUSIVITY_U: f64 = 2e-5;
/// Diffusivity of v (autocatalyst).
const DIFFUSIVITY_V: f64 = 1e-5;

/// Context slot of the Laplacian of column `component` (0: u, 1: v).
fn laplacian_variable(component: usize) -> ContextVariable {
    ContextVariable::External {
        name: if component == 0 {
            "laplacian_u".into()
        } else {
            "laplacian_v".into()
        },
    }
}

/// Gray–Scott on a ring, from a given initial state.
struct GrayScott {
    feed: f64,
    kill: f64,
    spacing: f64,
    initial: DMatrix<f64>,
}

impl RequiresContext for GrayScott {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![laplacian_variable(0), laplacian_variable(1)]
    }
}

impl PhysicalModel for GrayScott {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let s = state.as_vector_field()?;
        let n = s.nrows();
        let laplacians = [
            ctx.external(laplacian_variable(0))?.as_scalar_field()?,
            ctx.external(laplacian_variable(1))?.as_scalar_field()?,
        ];
        // Interior nodes from the calculators; the two end nodes across the
        // periodic closure (see the module documentation).
        let laplacian = |i: usize, j: usize| {
            if i == 0 || i == n - 1 {
                let (left, right) = ((i + n - 1) % n, (i + 1) % n);
                (s[(left, j)] - 2.0 * s[(i, j)] + s[(right, j)]) / (self.spacing * self.spacing)
            } else {
                laplacians[j][i]
            }
        };
        let mut rates = DMatrix::zeros(n, 2);
        for i in 0..n {
            let (u, v) = (s[(i, 0)], s[(i, 1)]);
            let reaction = u * v * v;
            rates[(i, 0)] = DIFFUSIVITY_U * laplacian(i, 0) - reaction + self.feed * (1.0 - u);
            rates[(i, 1)] =
                DIFFUSIVITY_V * laplacian(i, 1) + reaction - (self.feed + self.kill) * v;
        }
        Ok(ContextValue::VectorField(rates))
    }

    fn initial_state(&self, _mesh: &dyn Mesh) -> ContextValue {
        ContextValue::VectorField(self.initial.clone())
    }

    fn name(&self) -> &str {
        "gray_scott"
    }
}

/// The ring of `nodes` nodes: x_i = i * 2.5 / nodes, the last node one
/// spacing before the period.
fn ring(nodes: usize) -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(nodes, 0.0, PERIOD - PERIOD / nodes as f64)
}

/// Runs `model` on a ring of `nodes` nodes with RK4 up to `end_time`,
/// saving every `save_every` steps.
fn solve(
    model: GrayScott,
    nodes: usize,
    end_time: f64,
    steps: usize,
    save_every: usize,
) -> Result<SimulationResult, OxiflowError> {
    let mut config = SolverConfiguration::new(
        TimeConfiguration::new(
            end_time,
            StepControl::Fixed {
                dt: end_time / steps as f64,
            },
        )
        .saving_every(save_every),
        IntegratorKind::RK4,
    );
    for component in 0..2 {
        let mesh: Arc<dyn Mesh> = Arc::new(ring(nodes)?);
        config = config.with_calculator(Box::new(
            FDLaplacianCalculator::new(mesh, laplacian_variable(component))
                .with_component(component),
        ));
    }
    let scenario = Scenario::single(Box::new(model), Box::new(ring(nodes)?));
    RK4Solver.solve(&scenario, &config)
}

/// Fails with the quantity, the error and the tolerance.
fn check(quantity: &str, error: f64, tolerance: f64) -> Result<(), Box<dyn Error>> {
    if error.abs() <= tolerance {
        Ok(())
    } else {
        Err(format!("{quantity}: error {error:+.3e} exceeds the tolerance {tolerance:.1e}").into())
    }
}

// ── Case 1: Turing instability ────────────────────────────────────────────

const TURING_FEED: f64 = 0.066;
const TURING_KILL: f64 = 0.06245;
const TURING_NODES: usize = 256;
const TURING_END_TIME: f64 = 300.0;
const TURING_STEPS: usize = 300;
const MODE_AMPLITUDE: f64 = 1e-7;
const MODES: [usize; 4] = [5, 16, 31, 40];
const GROWTH_RATE_TOLERANCE: f64 = 1e-7;

/// Upper homogeneous state (u*, v*).
fn homogeneous_state() -> (f64, f64) {
    let (f, removal) = (TURING_FEED, TURING_FEED + TURING_KILL);
    let v = (f + (f * f - 4.0 * removal * removal * f).sqrt()) / (2.0 * removal);
    (removal / v, v)
}

/// Largest eigenvalue of J - q² diag(Du, Dv) at the homogeneous state, and
/// its eigenvector normalised to unit length; an error if the eigenvalues
/// are complex.
fn mode_growth(q2: f64) -> Result<(f64, [f64; 2]), Box<dyn Error>> {
    let (u, v) = homogeneous_state();
    let a = -v * v - TURING_FEED - DIFFUSIVITY_U * q2;
    let b = -2.0 * u * v;
    let c = v * v;
    let d = 2.0 * u * v - (TURING_FEED + TURING_KILL) - DIFFUSIVITY_V * q2;
    let half_trace = 0.5 * (a + d);
    let discriminant = half_trace * half_trace - (a * d - b * c);
    if discriminant < 0.0 {
        return Err(format!("complex eigenvalues for q² = {q2}").into());
    }
    let sigma = half_trace + discriminant.sqrt();
    let norm = b.hypot(sigma - a);
    Ok((sigma, [b / norm, (sigma - a) / norm]))
}

/// Projection of v - v* on cos(2 pi m i / n).
fn mode_amplitude(state: &DMatrix<f64>, mode: usize, v_star: f64) -> f64 {
    let n = state.nrows();
    (0..n)
        .map(|i| (state[(i, 1)] - v_star) * (2.0 * PI * (mode * i) as f64 / n as f64).cos())
        .sum::<f64>()
        * 2.0
        / n as f64
}

/// Measured and exact growth rates of `mode`, and the rate of the
/// continuous equations.
fn turing_mode(mode: usize) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let n = TURING_NODES;
    let dx = PERIOD / n as f64;
    let discrete_q2 = 4.0 / (dx * dx) * (PI * mode as f64 / n as f64).sin().powi(2);
    let continuous_q2 = (2.0 * PI * mode as f64 / PERIOD).powi(2);
    let (sigma, eigenvector) = mode_growth(discrete_q2)?;
    let (continuous_sigma, _) = mode_growth(continuous_q2)?;

    let (u_star, v_star) = homogeneous_state();
    let mut initial = DMatrix::zeros(n, 2);
    for i in 0..n {
        let shape = MODE_AMPLITUDE * (2.0 * PI * (mode * i) as f64 / n as f64).cos();
        initial[(i, 0)] = u_star + eigenvector[0] * shape;
        initial[(i, 1)] = v_star + eigenvector[1] * shape;
    }
    let start = mode_amplitude(&initial, mode, v_star);
    let model = GrayScott {
        feed: TURING_FEED,
        kill: TURING_KILL,
        spacing: dx,
        initial,
    };
    let result = solve(model, n, TURING_END_TIME, TURING_STEPS, TURING_STEPS)?;
    let last = result
        .states
        .last()
        .ok_or("no saved state")?
        .as_vector_field()?;
    let measured = (mode_amplitude(last, mode, v_star) / start).ln() / TURING_END_TIME;
    Ok((measured, sigma, continuous_sigma))
}

// ── Case 2: self-replicating pulses ───────────────────────────────────────

const PULSE_FEED: f64 = 0.04;
const PULSE_KILL: f64 = 0.06;
const PULSE_NODES: usize = 512;
const PULSE_END_TIME: f64 = 2500.0;
const PULSE_STEPS: usize = 5000;
/// Steps between saved states (every 100 time units).
const PULSE_SAVE_EVERY: usize = 200;
/// Half-width of the initial perturbation.
const PERTURBATION_HALF_WIDTH: f64 = 0.05;
/// A pulse is a run of nodes where v exceeds this level.
const PULSE_LEVEL: f64 = 0.2;
/// Expected pulse counts.
const PULSE_COUNTS: [(f64, usize); 3] = [(600.0, 2), (1500.0, 4), (2500.0, 6)];
const SYMMETRY_TOLERANCE: f64 = 1e-12;

/// Number of runs of consecutive nodes, around the ring, where v exceeds
/// `PULSE_LEVEL`.
fn pulse_count(state: &DMatrix<f64>) -> usize {
    let n = state.nrows();
    (0..n)
        .filter(|&i| state[(i, 1)] > PULSE_LEVEL && state[((i + n - 1) % n, 1)] <= PULSE_LEVEL)
        .count()
}

/// Largest difference between nodes i and n - i, over u and v.
fn asymmetry(state: &DMatrix<f64>) -> f64 {
    let n = state.nrows();
    (1..n)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| (state[(i, j)] - state[(n - i, j)]).abs())
        .fold(0.0, f64::max)
}

fn pulses() -> Result<SimulationResult, Box<dyn Error>> {
    let n = PULSE_NODES;
    let dx = PERIOD / n as f64;
    let middle = n / 2;
    let mut initial = DMatrix::zeros(n, 2);
    for i in 0..n {
        let inside = (i.abs_diff(middle) as f64) * dx < PERTURBATION_HALF_WIDTH;
        initial[(i, 0)] = if inside { 0.5 } else { 1.0 };
        initial[(i, 1)] = if inside { 0.25 } else { 0.0 };
    }
    let model = GrayScott {
        feed: PULSE_FEED,
        kill: PULSE_KILL,
        spacing: dx,
        initial,
    };
    Ok(solve(
        model,
        n,
        PULSE_END_TIME,
        PULSE_STEPS,
        PULSE_SAVE_EVERY,
    )?)
}

/// The state saved at `time`, or an error naming it.
fn state_at(result: &SimulationResult, time: f64) -> Result<&DMatrix<f64>, Box<dyn Error>> {
    let index = result
        .times
        .iter()
        .position(|&t| (t - time).abs() <= 1e-9 * time)
        .ok_or_else(|| format!("no state saved at t = {time}"))?;
    Ok(result.states[index].as_vector_field()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (u_star, v_star) = homogeneous_state();
    println!(
        "Case 1: Turing instability, F = {TURING_FEED}, k = {TURING_KILL}, u* = {u_star:.5}, v* = {v_star:.5}, {TURING_NODES} nodes, RK4, dt = {}, t = {TURING_END_TIME}",
        TURING_END_TIME / TURING_STEPS as f64
    );
    for mode in MODES {
        let (measured, exact, continuous) = turing_mode(mode)?;
        let error = (measured - exact) / exact;
        println!(
            "  mode {mode:>2}: sigma measured {measured:+.8e}, exact {exact:+.8e} (relative error {error:+.2e}); continuous equations {continuous:+.5e} (reported)"
        );
        check(
            &format!("growth rate of mode {mode} (relative)"),
            error,
            GROWTH_RATE_TOLERANCE,
        )?;
    }

    println!(
        "Case 2: self-replicating pulses, F = {PULSE_FEED}, k = {PULSE_KILL}, {PULSE_NODES} nodes, RK4, dt = {}",
        PULSE_END_TIME / PULSE_STEPS as f64
    );
    let result = pulses()?;
    let history: Vec<String> = result
        .states
        .iter()
        .zip(&result.times)
        .map(|(state, t)| Ok(format!("{t}:{}", pulse_count(state.as_vector_field()?))))
        .collect::<Result<_, OxiflowError>>()?;
    println!("  pulse counts (t:count): {} (reported)", history.join(" "));
    let mut largest_asymmetry: f64 = 0.0;
    for state in &result.states {
        largest_asymmetry = largest_asymmetry.max(asymmetry(state.as_vector_field()?));
    }
    println!("  largest mirror asymmetry: {largest_asymmetry:.1e}");
    for (time, expected) in PULSE_COUNTS {
        let count = pulse_count(state_at(&result, time)?);
        if count != expected {
            return Err(format!("pulse count at t = {time}: {count}, expected {expected}").into());
        }
    }
    println!("  pulse counts at the checked times as expected");
    check("mirror symmetry", largest_asymmetry, SYMMETRY_TOLERANCE)?;
    println!("gray_scott_1d: every check within its tolerance");
    Ok(())
}
