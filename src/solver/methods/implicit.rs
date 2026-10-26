//! # Module `solver::methods::implicit`
//!
//! Shared machinery for implicit (theta-method) integrators — DD-033
//! ([#86](https://github.com/biface/oxiflow/issues/86)).
//!
//! [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver) (θ=1),
//! [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
//! (θ=0.5) and [`BDF2Solver`](super::bdf2::BDF2Solver) are thin wrappers
//! around `implicit_step_newton` (DD-044,
//! [#111](https://github.com/biface/oxiflow/issues/111),
//! [#112](https://github.com/biface/oxiflow/issues/112)) and, with the
//! `sparse` feature, `implicit_step_adaptive` (DD-043). Each solver names
//! its implicit equation with an `ImplicitFormula` — the theta method or
//! BDF2 — and every formula runs the shared [`super::newton`] loop; the
//! two paths differ only in the storage format of the linear system, dense
//! or banded.
//!
//! ## Why one theta formula serves two methods
//!
//! The generalised theta method is:
//!
//! $$u^{n+1} = u^n + \Delta t\left[(1-\theta) f(u^n) + \theta f(u^{n+1})\right]$$
//!
//! Linearising the residual $g(u) = u - u^n - \Delta t[(1-\theta)f(u^n) +
//! \theta f(u)]$ at $u = u^n$ gives $g(u^n) = -\Delta t \cdot f(u^n)$ for
//! **any** $\theta$ — the right-hand side of the linear correction doesn't
//! depend on θ at all. Only the system matrix does
//! ($I - \theta \Delta t J_f$). One function, one parameter.
//!
//! ## Iterated Newton, single correction by default (DD-033, DD-044)
//!
//! DD-033 ([#86](https://github.com/biface/oxiflow/issues/86)) started with
//! exactly **one** Newton-style correction, with the Jacobian frozen at
//! $u^n$ — exact when `compute_physics` is affine in `u`, a first-order
//! approximation otherwise. DD-044
//! ([#111](https://github.com/biface/oxiflow/issues/111)) replaced it with
//! the iterated [`super::newton`] loop, configured by the solver's
//! `with_newton_convergence`/`with_jacobian_strategy`/
//! `with_max_newton_iterations` builders. Their defaults
//! (`NewtonParams::default()`) keep a single frozen-Jacobian correction,
//! which reproduces the historical output exactly on affine problems (see
//! [`super::newton`]'s own docs for why no special-casing is needed), and
//! report [`OxiflowError::NewtonNotConverged`] rather than a first-order
//! approximation on a nonlinear one.
//!
//! The dense and the banded paths share the loop, the residual and the
//! configuration, whatever the formula: the banded path assembles
//! $\alpha I - c J_f$ in compressed columns from
//! `banded_finite_difference_jacobian` and solves it with the solver's
//! `SparseLinearSolver`, nothing more.
//!
//! ## State variants
//!
//! Newton and both finite-difference Jacobians work on the flat vector of
//! unknowns given by `StateLayout` (`super::state_layout`): a `ScalarField`
//! as is, a `VectorField` interleaved by node, so a `VectorField` state is
//! advanced like a `ScalarField` (#192). The banded path therefore takes a
//! half-bandwidth counted in flat unknowns, `d·(s + 1) − 1` for `d`
//! components and a stencil of half-width `s`.
//!
//! ## Boundary conditions × the perturbed Jacobian
//!
//! `finite_difference_jacobian` perturbs each component of the state and
//! re-evaluates `evaluate_derivative`, which re-applies boundary
//! conditions on every call. A Dirichlet-constrained node will have its
//! perturbation overwritten before `compute_physics` sees it — physically
//! correct (a BC-constrained node isn't a free unknown): the constrained
//! node's own *column* in the assembled Jacobian is exactly zero, while
//! its *row* still reflects genuine physics-based coupling to
//! unconstrained neighbours. Verified by
//! `tests::dirichlet_constrained_column_is_exactly_zero_row_keeps_real_coupling`
//! (#113) — previously documented as a known untested limitation (DD-033,
//! [#86](https://github.com/biface/oxiflow/issues/86), v0.4.0).
//!
//! ## Parallel assembly (DT-069, #198)
//!
//! With the `parallel` feature, both finite-difference Jacobians dispatch
//! their independent evaluations through Rayon — one per column for the
//! dense one, one per color for the banded one — once the system has at
//! least as many unknowns as the solver's dispatch threshold. Each Rayon
//! job perturbs its own work state and writes its columns in place, in the
//! final matrix. Every column is computed from the same operands with the
//! same operations in both modes, so the sequential and parallel Jacobians
//! are bit-for-bit identical. The threshold belongs to each implicit
//! solver (`JacobianDispatch`); without the feature it does not exist and
//! the assembly is sequential.

use nalgebra::{DMatrix, DVector};

use super::evaluate_derivative;
use super::newton::{self, JacobianStrategy, NewtonConvergence, NewtonParams, NewtonSystem};
use super::state_layout::StateLayout;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::linear::LinearSolver;
#[cfg(feature = "parallel")]
use crate::solver::parallel::ParallelThreshold;
use crate::solver::scenario::Domain;
#[cfg(feature = "sparse")]
use crate::solver::sparse::SparseLinearSolver;
#[cfg(feature = "sparse")]
use faer::sparse::{SparseColMat, SymbolicSparseColMat};

/// Finite-difference step size for [`finite_difference_jacobian`].
///
/// Not scaled by state magnitude (v1 simplification) — fine for the O(1)
/// stiff linear test problems this targets; revisit if used on
/// significantly different magnitude scales.
const FD_EPSILON: f64 = 1e-7;

/// Default Rayon dispatch threshold of the Jacobian assembly, in unknowns
/// (DT-069, #198) — an indicative value, shared by the three implicit
/// solvers since they share this assembly. Each dispatched task is a full
/// derivative evaluation, much heavier than one stencil point, so the
/// crossover is expected far below
/// [`crate::operators::fd::default_parallel_threshold`] (49 999). Measure it
/// on the target workload with the `oxiflow_parallel` benchmarks and set
/// it through the solver's `with_parallel_threshold`.
#[cfg(feature = "parallel")]
pub(crate) fn default_parallel_threshold() -> usize {
    256
}

/// Dispatch configuration of the Jacobian assembly, held by each implicit
/// solver and passed down to [`finite_difference_jacobian`] and, with the
/// `sparse` feature, `banded_finite_difference_jacobian` (DT-069, #198).
///
/// Holds a `ParallelThreshold` only with the `parallel` feature: without
/// it the type is empty, no threshold is representable, and the assembly
/// is sequential. Passing this configuration rather than a bare number
/// keeps the threshold out of the feature-agnostic signatures, without one
/// version of every θ-method function per feature.
#[derive(Debug, Default)]
pub(crate) struct JacobianDispatch {
    #[cfg(feature = "parallel")]
    threshold: ParallelThreshold,
}

impl JacobianDispatch {
    /// Whether the assembly of a system of `n` unknowns goes through Rayon:
    /// `n >= threshold`, the comparison used by every dispatch site of the
    /// crate. Always `false` without the `parallel` feature.
    pub(crate) fn is_parallel_for(&self, n: usize) -> bool {
        #[cfg(feature = "parallel")]
        {
            n >= self.threshold()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = n;
            false
        }
    }

    /// Effective threshold: the override if one was set, otherwise
    /// [`default_parallel_threshold`].
    #[cfg(feature = "parallel")]
    pub(crate) fn threshold(&self) -> usize {
        self.threshold.get(default_parallel_threshold())
    }

    /// Overrides the threshold.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — see `ParallelThreshold::set`.
    #[cfg(feature = "parallel")]
    pub(crate) fn set(&self, threshold: usize) {
        self.threshold.set(threshold);
    }
}

/// Runs `fill` on every item, each call with a work state cloned from
/// `state` that `fill` refills and perturbs itself.
///
/// Through Rayon when `parallel`, with one work state per Rayon job
/// (`try_for_each_init` initialises once per job, not once per thread):
/// of the order of the number of threads, not of items;
/// otherwise in order, with one work state for all items. Items must be
/// independent of one another: each writes only its own columns, from
/// its own work state, so the result does not depend on the dispatch.
fn try_for_each_with_work<T, F>(
    items: Vec<T>,
    parallel: bool,
    state: &ContextValue,
    fill: F,
) -> Result<(), OxiflowError>
where
    T: Send,
    F: Fn(&mut ContextValue, T) -> Result<(), OxiflowError> + Sync,
{
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return items
            .into_par_iter()
            .try_for_each_init(|| state.clone(), |work, item| fill(work, item));
    }
    #[cfg(not(feature = "parallel"))]
    debug_assert!(
        !parallel,
        "no parallel dispatch without the `parallel` feature"
    );

    let mut work = state.clone();
    items.into_iter().try_for_each(|item| fill(&mut work, item))
}

/// Estimates $\partial f/\partial u$ at `state`, `t`, by forward
/// differences.
///
/// For `f` affine in `u` (the case these implicit methods are validated
/// against), forward differences are exact regardless of step size — no
/// truncation error from the linear approximation itself.
///
/// Rows and columns follow the flat ordering of [`StateLayout`]: for a
/// `VectorField` state, unknowns are interleaved by node (#192).
///
/// A work state, cloned from `state`, is refilled before each column and
/// perturbed in place (#197): the boundary conditions modify the evaluated
/// state in place and do not report which nodes they changed, so a full
/// refill per column is needed, but no allocation is. Columns are
/// dispatched according to `dispatch` — see
/// [parallel assembly](self#parallel-assembly-dt-069-198).
///
/// See [boundary conditions × the perturbed Jacobian](self#boundary-conditions--the-perturbed-jacobian).
pub(crate) fn finite_difference_jacobian(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &ContextValue,
    t: f64,
    dt: f64,
    dispatch: &JacobianDispatch,
) -> Result<DMatrix<f64>, OxiflowError> {
    let layout = StateLayout::of(state)?;
    let base = layout.values(state)?;
    let n = layout.len();

    let mut base_state = state.clone();
    let f0 = evaluate_derivative(domain, chain, &mut base_state, t, dt)?;
    let f0_field = layout.flatten(&f0)?.into_owned();

    let mut jacobian = DMatrix::<f64>::zeros(n, n);
    if n == 0 {
        return Ok(jacobian);
    }

    // Column-major storage: column j is the j-th chunk of n values.
    let columns: Vec<(usize, &mut [f64])> =
        jacobian.as_mut_slice().chunks_mut(n).enumerate().collect();

    try_for_each_with_work(
        columns,
        dispatch.is_parallel_for(n),
        state,
        |work, (j, column)| {
            let values = layout.values_mut(work)?;
            values.copy_from_slice(base);
            values[layout.storage_index(j)] += FD_EPSILON;

            let f_j = evaluate_derivative(domain, chain, work, t, dt)?;
            let f_j_values = layout.values(&f_j)?;

            for (i, entry) in column.iter_mut().enumerate() {
                *entry = (f_j_values[layout.storage_index(i)] - f0_field[i]) / FD_EPSILON;
            }
            Ok(())
        },
    )?;

    Ok(jacobian)
}

/// Convergence criterion guaranteeing that the single-correction default of
/// the implicit solvers succeeds on the stiffest affine problems
/// this crate currently tests (`λ = 1e4`-scale exponential decay).
///
/// Deliberately distinct from [`NewtonConvergence::default`] (`tol_abs =
/// 1e-8`): on such problems, [`finite_difference_jacobian`]'s fixed step
/// size suffers catastrophic-cancellation error whose absolute magnitude
/// scales roughly as `λ · ε_machine / FD_EPSILON` — empirically up to
/// ~1e-6 for the stiffest cases tested. The single correction itself is
/// still mathematically exact for affine `f`; this is a
/// finite-difference precision floor, not a nonlinearity signal, and
/// [`NewtonConvergence::default`] remains the right general-purpose
/// starting point for callers configuring the loop themselves — it just
/// isn't loose enough to guarantee success on this specific,
/// already-tested edge of the stiffness range. Used here and by
/// [`BackwardEulerSolver::default`](super::backward_euler::BackwardEulerSolver)/
/// [`CrankNicolsonSolver::default`](super::crank_nicolson::CrankNicolsonSolver)
/// — not by `NewtonConvergence`'s own `Default` impl — precisely so that
/// zero-config solver construction keeps reproducing history exactly,
/// without changing what a fresh, explicitly-configured
/// `NewtonConvergence::default()` means elsewhere.
pub(crate) fn stiff_jacobian_convergence() -> NewtonConvergence {
    NewtonConvergence::ResidualOnly {
        tol_abs: 1e-5,
        tol_rel: 1e-6,
    }
}

/// Zero-config Newton parameters of the implicit solvers
/// ([`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver),
/// [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver),
/// [`BDF2Solver`](super::bdf2::BDF2Solver)): [`stiff_jacobian_convergence`],
/// [`JacobianStrategy::ModifiedFrozen`] and one iteration, the pre-DD-044
/// single frozen-Jacobian correction. It is exact on affine problems (see
/// [`newton::solve`]'s docs); a genuinely nonlinear model needs a larger
/// budget, typically with [`JacobianStrategy::FullNewton`].
impl Default for NewtonParams {
    fn default() -> Self {
        Self {
            convergence: stiff_jacobian_convergence(),
            jacobian_strategy: JacobianStrategy::default(),
            max_iterations: 1,
        }
    }
}

/// Implicit step formula: how the implicit equation of an integrator is
/// written in [`newton`]'s residual convention $g(u) = \alpha (u - u^{*}) - c
/// f(u)$. Only the triple $(u^{*}, \alpha, c)$ differs between the
/// integrators; the Newton loop, the residual, the Jacobians and the
/// choice of storage format are shared (see [`implicit_step_newton`] and,
/// with the `sparse` feature, `implicit_step_adaptive`).
#[derive(Debug, Clone, Copy)]
pub(crate) enum ImplicitFormula<'a> {
    /// Generalised theta method: $\alpha = 1$, $c = \theta \Delta t$,
    /// $u^{*} = u^n + \Delta t(1-\theta)f(u^n)$. `θ = 1` is Backward Euler,
    /// `θ = 0.5` Crank-Nicolson.
    Theta(f64),
    /// BDF2 with fixed step: $\alpha = 3/2$, $c = \Delta t$, $u^{*} =
    /// \frac{4u^n - u^{n-1}}{3}$, derived from $g(u) = \frac{3}{2}u - 2u^n +
    /// \frac{1}{2}u^{n-1} - \Delta t f(u)$. `previous` is $u^{n-1}$ and must
    /// have the layout of $u^n$.
    Bdf2 {
        /// The state $u^{n-1}$.
        previous: &'a ContextValue,
    },
}

impl ImplicitFormula<'_> {
    /// Returns $(u^{*}, \alpha, c)$ for the step from `u_n`, given
    /// $f(u^n)$ (`f_n`) and the time step.
    ///
    /// # Errors
    ///
    /// Returns [`OxiflowError::InvalidDomain`] (prefixed with `history
    /// state:`) if the BDF2 `previous` state does not have the layout of
    /// `u_n`, and propagates the flattening errors of `f_n`.
    fn residual_terms(
        &self,
        layout: &StateLayout,
        u_n: DVector<f64>,
        f_n: &ContextValue,
        dt: f64,
    ) -> Result<(DVector<f64>, f64, f64), OxiflowError> {
        match *self {
            Self::Theta(theta) => {
                let f_n = layout.flatten(f_n)?.into_owned();
                Ok((u_n + f_n * (dt * (1.0 - theta)), 1.0, theta * dt))
            }
            Self::Bdf2 { previous } => {
                let u_prev = layout
                    .flatten(previous)
                    .map_err(|e| match e {
                        OxiflowError::InvalidDomain(message) => {
                            OxiflowError::InvalidDomain(format!("history state: {message}"))
                        }
                        other => other,
                    })?
                    .into_owned();
                Ok(((u_n * 4.0 - u_prev) / 3.0, 1.5, dt))
            }
        }
    }
}

/// Performs one implicit step of `formula` through the iterated Newton
/// loop ([`newton::solve`]) with the dense linear system, under the
/// caller's Newton parameters (DD-044, #111, #112).
///
/// Every implicit solver calls this
/// ([`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver),
/// [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver),
/// [`BDF2Solver`](super::bdf2::BDF2Solver) and its bootstrap step) when
/// no banded path is configured, with the solver's own configuration and
/// Jacobian dispatch (DT-069, #198).
///
/// `state` is mutated in-place by boundary condition application — same
/// contract as the explicit solvers (see
/// [`crate::solver::methods::evaluate_derivative`]): callers should not
/// assume `state` is left unchanged after this call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn implicit_step_newton(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    formula: ImplicitFormula<'_>,
    linear_solver: &dyn LinearSolver,
    newton: &NewtonParams,
    dispatch: &JacobianDispatch,
) -> Result<ContextValue, OxiflowError> {
    implicit_newton(
        domain,
        chain,
        state,
        t,
        dt,
        formula,
        newton,
        |layout, t_target| {
            newton::DenseNewtonSystem::new(
                move |candidate: &DVector<f64>| {
                    let candidate_state = layout.rebuild(candidate.clone())?;
                    finite_difference_jacobian(
                        domain,
                        chain,
                        &candidate_state,
                        t_target,
                        dt,
                        dispatch,
                    )
                },
                linear_solver,
            )
        },
    )
}

/// The implicit step common to every formula and every storage format:
/// builds the residual of `formula` and runs [`newton::solve`] with the
/// [`NewtonSystem`] that `system` returns for the state layout and the
/// target time `t + dt`.
///
/// `f(u^n)` is evaluated once, at time `t` — it applies the boundary
/// conditions to `state` in every case, and is the known term of the
/// theta method (BDF2 does not use it); every subsequent candidate
/// `f`/Jacobian evaluation inside the Newton loop is at time `t + dt`
/// (the implicit/target time).
#[allow(clippy::too_many_arguments)]
fn implicit_newton<S: NewtonSystem>(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    formula: ImplicitFormula<'_>,
    newton: &NewtonParams,
    system: impl FnOnce(StateLayout, f64) -> S,
) -> Result<ContextValue, OxiflowError> {
    // f(u^n, t) -- BCs applied in-place to `state` itself, consistent with
    // the explicit solvers' contract.
    let f_n = evaluate_derivative(domain, chain, state, t, dt)?;
    // Newton works on the flat vector of unknowns (#192).
    let layout = StateLayout::of(state)?;
    let u_n_field = layout.flatten(state)?.into_owned();

    let u0 = u_n_field.clone();
    let (u_star, alpha, coeff) = formula.residual_terms(&layout, u_n_field, &f_n, dt)?;
    let t_target = t + dt;

    let u_next = newton::solve(
        u0,
        &u_star,
        alpha,
        coeff,
        move |candidate: &DVector<f64>| {
            let mut candidate_state = layout.rebuild(candidate.clone())?;
            let f = evaluate_derivative(domain, chain, &mut candidate_state, t_target, dt)?;
            Ok(layout.flatten(&f)?.into_owned())
        },
        &mut system(layout, t_target),
        newton,
    )?;

    layout.rebuild(u_next)
}

// ── Sparse / banded path (DD-013 second phase, DD-043) ─────────────────────────
//
// `#[cfg(feature = "sparse")]` only. Bandwidth is always an explicit
// parameter here, supplied by the caller (the solver's own HOW-side
// configuration, e.g. `BackwardEulerSolver::with_jacobian_bandwidth`) — it
// is never read from `Domain`/`PhysicalModel`. See DD-043, amendment 1, for
// why `PhysicalModel::jacobian_bandwidth` was rejected: bandwidth is a
// property of the numerical scheme (HOW), not of the physical model (WHAT).

/// Assembles $\partial f/\partial u$ within a band of half-width
/// `bandwidth` around the diagonal, directly in compressed-column form, in
/// the flat ordering of [`StateLayout`] (#192): for a `VectorField` of d
/// components and a stencil of half-width `s`, the band half-width is
/// `d·(s + 1) − 1`.
///
/// Columns are grouped by Curtis-Powell-Reid (CPR) coloring: for a
/// genuinely banded Jacobian, columns `2*bandwidth + 1` or more apart never
/// share a nonzero row, so all columns of the same color are perturbed
/// **simultaneously** in a single [`evaluate_derivative`] call —
/// `2*bandwidth + 1` evaluations, instead of [`finite_difference_jacobian`]'s
/// `n`. As there, a work state is refilled and perturbed in place for each
/// color (#197), and colors are dispatched according to `dispatch` — the
/// threshold is compared with the number of unknowns `n`, not of colors.
///
/// The structure is known in advance (DT-069, #198): column `j` holds rows
/// `max(0, j − bandwidth)` to `min(n, j + bandwidth + 1)`, in increasing
/// order, diagonal included. `col_ptr` and `row_idx` are built from `n` and
/// `bandwidth`; each column of `values` is written once, in place, with the
/// same arithmetic as [`finite_difference_jacobian`] — no triplets, no
/// sort. Every entry of the band is stored, zeros included.
#[cfg(feature = "sparse")]
pub(crate) fn banded_finite_difference_jacobian(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &ContextValue,
    t: f64,
    dt: f64,
    bandwidth: usize,
    dispatch: &JacobianDispatch,
) -> Result<SparseColMat<usize, f64>, OxiflowError> {
    let layout = StateLayout::of(state)?;
    let base = layout.values(state)?;
    let n = layout.len();

    let mut base_state = state.clone();
    let f0 = evaluate_derivative(domain, chain, &mut base_state, t, dt)?;
    let f0_field = layout.flatten(&f0)?.into_owned();

    let rows_of_column = |j: usize| j.saturating_sub(bandwidth)..(j + bandwidth + 1).min(n);
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        row_idx.extend(rows_of_column(j));
        col_ptr.push(row_idx.len());
    }
    let mut values = vec![0.0; row_idx.len()];

    // One group of (column, column storage) per color; column `j` has
    // color `j % n_colors`, and colors from `n` on have no column.
    let n_colors = 2 * bandwidth + 1;
    let mut colors: Vec<Vec<(usize, &mut [f64])>> =
        (0..n_colors.min(n)).map(|_| Vec::new()).collect();
    let mut rest = values.as_mut_slice();
    for j in 0..n {
        let (column, tail) = std::mem::take(&mut rest).split_at_mut(col_ptr[j + 1] - col_ptr[j]);
        colors[j % n_colors].push((j, column));
        rest = tail;
    }

    try_for_each_with_work(colors, dispatch.is_parallel_for(n), state, |work, color| {
        let values = layout.values_mut(work)?;
        values.copy_from_slice(base);
        for (j, _) in &color {
            values[layout.storage_index(*j)] += FD_EPSILON;
        }

        let f_c = evaluate_derivative(domain, chain, work, t, dt)?;
        let f_c_values = layout.values(&f_c)?;

        // The bands of the columns of one color ([j-bandwidth,
        // j+bandwidth]) never overlap: each row's response is attributable
        // to exactly one column.
        for (j, column) in color {
            for (i, entry) in rows_of_column(j).zip(column.iter_mut()) {
                *entry = (f_c_values[layout.storage_index(i)] - f0_field[i]) / FD_EPSILON;
            }
        }
        Ok(())
    })?;

    let symbolic = SymbolicSparseColMat::new_checked(n, n, col_ptr, None, row_idx);
    Ok(SparseColMat::new(symbolic, values))
}

/// Banded [`NewtonSystem`] (DD-043): `jacobian` evaluates $J_f$ within the
/// band of half-width `bandwidth`, in compressed columns (in production,
/// [`banded_finite_difference_jacobian`]); $\alpha I - c J_f$ is folded in
/// place into its values, and `sparse_solver` solves the system.
#[cfg(feature = "sparse")]
struct BandedNewtonSystem<'a, J> {
    jacobian: J,
    bandwidth: usize,
    sparse_solver: &'a dyn SparseLinearSolver,
    matrix: Option<SparseColMat<usize, f64>>,
}

#[cfg(feature = "sparse")]
impl<J> NewtonSystem for BandedNewtonSystem<'_, J>
where
    J: FnMut(&DVector<f64>) -> Result<SparseColMat<usize, f64>, OxiflowError>,
{
    fn assemble(&mut self, u: &DVector<f64>, alpha: f64, coeff: f64) -> Result<(), OxiflowError> {
        let mut matrix = (self.jacobian)(u)?;
        // alpha*I - coeff*J, folded in place into the values of J. Column j
        // starts at row max(0, j - bandwidth) and always holds its diagonal
        // entry, at offset j - max(0, j - bandwidth).
        for j in 0..matrix.ncols() {
            let column = matrix.val_of_col_mut(j);
            for value in column.iter_mut() {
                *value *= -coeff;
            }
            column[j - j.saturating_sub(self.bandwidth)] += alpha;
        }
        self.matrix = Some(matrix);
        Ok(())
    }

    fn solve(&self, rhs: &DVector<f64>) -> Result<DVector<f64>, OxiflowError> {
        let matrix = self
            .matrix
            .as_ref()
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "BandedNewtonSystem::solve",
                message: "no matrix assembled".into(),
            })?;
        self.sparse_solver.solve(matrix, rhs)
    }
}

/// Same contract as [`implicit_step_newton`], with a banded path: when
/// `jacobian_bandwidth` is `Some(k)` **and** `n > sparse_threshold`, the
/// Newton loop runs over a banded system, the Jacobian assembled by
/// [`banded_finite_difference_jacobian`] and the corrections solved with
/// `sparse_solver`. Every other case — `None` bandwidth, or a small enough
/// system that the dense path is cheaper regardless of bandwidth — takes
/// [`implicit_step_newton`]. Both paths run the same iterated loop
/// with the caller's Newton parameters; they differ only in the storage
/// format of the linear system.
///
/// `jacobian_bandwidth`/`sparse_threshold`/`sparse_solver`/`dispatch` are
/// supplied by the caller's own HOW-side configuration (DD-043, amendment
/// 1; DT-069) — never read from `Domain` or `PhysicalModel`. `dispatch`
/// applies to both paths.
#[cfg(feature = "sparse")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn implicit_step_adaptive(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    formula: ImplicitFormula<'_>,
    dense_solver: &dyn LinearSolver,
    sparse_solver: &dyn SparseLinearSolver,
    sparse_threshold: usize,
    jacobian_bandwidth: Option<usize>,
    newton: &NewtonParams,
    dispatch: &JacobianDispatch,
) -> Result<ContextValue, OxiflowError> {
    let n = StateLayout::of(state)?.len();

    match jacobian_bandwidth {
        Some(bandwidth) if n > sparse_threshold => implicit_newton(
            domain,
            chain,
            state,
            t,
            dt,
            formula,
            newton,
            |layout, t_target| BandedNewtonSystem {
                jacobian: move |candidate: &DVector<f64>| {
                    let candidate_state = layout.rebuild(candidate.clone())?;
                    banded_finite_difference_jacobian(
                        domain,
                        chain,
                        &candidate_state,
                        t_target,
                        dt,
                        bandwidth,
                        dispatch,
                    )
                },
                bandwidth,
                sparse_solver,
                matrix: None,
            },
        ),
        _ => implicit_step_newton(
            domain,
            chain,
            state,
            t,
            dt,
            formula,
            dense_solver,
            newton,
            dispatch,
        ),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::model::ExponentialDecay;
    use crate::solver::chain::build_calculator_chain;
    use crate::solver::linear::NalgebraDenseSolver;
    use crate::solver::scenario::Scenario;

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    /// Two-component `VectorField` model: a reaction matrix couples the
    /// components of each node, and each component diffuses to its
    /// neighbours (zero outside the domain). Affine in `u`. In the
    /// node-interleaved order its Jacobian has half-bandwidth
    /// `d·(s + 1) − 1 = 3` (d = 2, s = 1). Also used by `sparse_tests`.
    #[derive(Debug)]
    pub(super) struct CoupledReactionDiffusion {
        pub(super) reaction: [[f64; 2]; 2],
        pub(super) diffusion: f64,
    }

    impl CoupledReactionDiffusion {
        pub(super) const COMPONENTS: usize = 2;

        /// Exact Jacobian entry between unknown `(node, component)` pairs.
        fn exact_entry(&self, (i, c): (usize, usize), (m, k): (usize, usize)) -> f64 {
            let reaction = if m == i { self.reaction[c][k] } else { 0.0 };
            let diffusion = match (k == c, i.abs_diff(m)) {
                (true, 0) => -2.0 * self.diffusion,
                (true, 1) => self.diffusion,
                _ => 0.0,
            };
            reaction + diffusion
        }
    }

    impl RequiresContext for CoupledReactionDiffusion {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for CoupledReactionDiffusion {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_vector_field()?;
            let n = u.nrows();
            let out = DMatrix::from_fn(n, Self::COMPONENTS, |i, c| {
                let reaction = self.reaction[c][0] * u[(i, 0)] + self.reaction[c][1] * u[(i, 1)];
                let left = if i > 0 { u[(i - 1, c)] } else { 0.0 };
                let right = if i + 1 < n { u[(i + 1, c)] } else { 0.0 };
                reaction + self.diffusion * (left - 2.0 * u[(i, c)] + right)
            });
            Ok(ContextValue::VectorField(out))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::VectorField(DMatrix::from_fn(mesh.n_dof(), Self::COMPONENTS, |i, c| {
                1.0 + 0.1 * i as f64 - 0.3 * c as f64
            }))
        }

        fn name(&self) -> &str {
            "coupled_reaction_diffusion_test"
        }
    }

    pub(super) fn coupled_reaction_diffusion() -> CoupledReactionDiffusion {
        CoupledReactionDiffusion {
            reaction: [[-1.5, 0.4], [0.7, -0.9]],
            diffusion: 0.6,
        }
    }

    #[test]
    fn jacobian_of_a_vector_field_follows_the_node_interleaved_order() {
        let nodes = 5;
        let d = CoupledReactionDiffusion::COMPONENTS;
        let model = coupled_reaction_diffusion();
        let expected =
            |row: usize, col: usize| model.exact_entry((row / d, row % d), (col / d, col % d));
        let scenario = Scenario::single(Box::new(coupled_reaction_diffusion()), make_mesh(nodes));
        let domain = scenario.single_domain().unwrap();
        let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
        let state = domain.model.initial_state(domain.mesh.as_ref());

        let jac = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.1,
            &JacobianDispatch::default(),
        )
        .unwrap();

        assert_eq!(jac.shape(), (nodes * d, nodes * d));
        for row in 0..nodes * d {
            for col in 0..nodes * d {
                let exact = expected(row, col);
                if exact == 0.0 {
                    // The perturbed unknown does not enter this row: same
                    // inputs, same operations, hence exactly zero.
                    assert_eq!(jac[(row, col)], 0.0, "({row}, {col})");
                } else {
                    assert!(
                        (jac[(row, col)] - exact).abs() < 1e-6,
                        "({row}, {col}): {} vs {exact}",
                        jac[(row, col)]
                    );
                }
            }
        }
    }

    // ── Jacobian dispatch (DT-069, #198) ─────────────────────────────────────

    /// Dispatch configuration with the given threshold.
    #[cfg(feature = "parallel")]
    pub(super) fn dispatch_with_threshold(threshold: usize) -> JacobianDispatch {
        let dispatch = JacobianDispatch::default();
        dispatch.set(threshold);
        dispatch
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn jacobian_dispatch_compares_the_number_of_unknowns_with_its_threshold() {
        let dispatch = JacobianDispatch::default();
        let default = default_parallel_threshold();
        assert_eq!(dispatch.threshold(), default);
        assert!(!dispatch.is_parallel_for(default - 1));
        assert!(dispatch.is_parallel_for(default));
        dispatch.set(3);
        assert_eq!(dispatch.threshold(), 3);
        assert!(!dispatch.is_parallel_for(2));
        assert!(dispatch.is_parallel_for(3));
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[should_panic(expected = "parallel threshold must be at least 1")]
    fn jacobian_dispatch_rejects_a_zero_threshold() {
        JacobianDispatch::default().set(0);
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn jacobian_dispatch_is_sequential_without_the_parallel_feature() {
        assert!(!JacobianDispatch::default().is_parallel_for(usize::MAX));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn dense_jacobian_is_bit_identical_in_sequential_and_parallel_dispatch() {
        let scenarios = [
            Scenario::single(Box::new(ExponentialDecay { lambda: 2.0 }), make_mesh(9)),
            Scenario::single(Box::new(coupled_reaction_diffusion()), make_mesh(6)),
        ];
        for scenario in &scenarios {
            let domain = scenario.single_domain().unwrap();
            let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
            let state = domain.model.initial_state(domain.mesh.as_ref());
            let jacobian = |threshold| {
                let dispatch = dispatch_with_threshold(threshold);
                finite_difference_jacobian(domain, &chain, &state, 0.0, 0.1, &dispatch).unwrap()
            };
            let (parallel, sequential) = (jacobian(1), jacobian(usize::MAX));
            assert_eq!(parallel.shape(), sequential.shape());
            assert!(parallel
                .iter()
                .zip(sequential.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    /// `du/dt = -u`, recording at each evaluation whether it ran on a Rayon
    /// worker thread.
    #[cfg(feature = "parallel")]
    #[derive(Debug)]
    struct WorkerRecorder {
        on_worker: std::sync::Arc<std::sync::Mutex<Vec<bool>>>,
    }

    #[cfg(feature = "parallel")]
    impl RequiresContext for WorkerRecorder {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    #[cfg(feature = "parallel")]
    impl PhysicalModel for WorkerRecorder {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            self.on_worker
                .lock()
                .unwrap()
                .push(rayon::current_thread_index().is_some());
            Ok(ContextValue::ScalarField(
                state.as_scalar_field()?.map(|v| -v),
            ))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }

        fn name(&self) -> &str {
            "worker_recorder"
        }
    }

    /// Runs `step` once on a domain of `n` unknowns and reports whether any
    /// derivative evaluation ran on a Rayon worker thread.
    #[cfg(feature = "parallel")]
    fn evaluated_on_a_worker(
        n: usize,
        step: impl Fn(
            &Domain,
            &[&dyn ContextCalculator],
            &mut ContextValue,
        ) -> Result<ContextValue, OxiflowError>,
    ) -> bool {
        let on_worker = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let model = WorkerRecorder {
            on_worker: std::sync::Arc::clone(&on_worker),
        };
        let scenario = Scenario::single(Box::new(model), make_mesh(n));
        let domain = scenario.single_domain().unwrap();
        let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        step(domain, &chain, &mut state).unwrap();
        let recorded = on_worker.lock().unwrap();
        assert!(!recorded.is_empty());
        recorded.iter().any(|&on_worker| on_worker)
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn implicit_solvers_hand_their_threshold_to_the_jacobian_assembly() {
        use crate::solver::methods::backward_euler::BackwardEulerSolver;
        use crate::solver::methods::bdf2::BDF2Solver;
        use crate::solver::methods::crank_nicolson::CrankNicolsonSolver;
        use crate::solver::methods::SteppableSolver;

        let n = 8;
        let history = [ContextValue::ScalarField(DVector::from_element(n, 1.0))];
        for threshold in [1, usize::MAX] {
            let parallel = threshold == 1;
            let be = BackwardEulerSolver::new().with_parallel_threshold(threshold);
            let cn = CrankNicolsonSolver::new().with_parallel_threshold(threshold);
            let bdf2 = BDF2Solver::new().with_parallel_threshold(threshold);
            let cases: [(&str, &dyn SteppableSolver, &[ContextValue]); 4] = [
                ("Backward Euler", &be, &[]),
                ("Crank-Nicolson", &cn, &[]),
                ("BDF2 startup", &bdf2, &[]),
                ("BDF2 update", &bdf2, &history),
            ];
            for (name, solver, history) in cases {
                let on_worker = evaluated_on_a_worker(n, |domain, chain, state| {
                    solver.step(domain, chain, state, history, 0.0, 0.1)
                });
                assert_eq!(on_worker, parallel, "{name}, threshold {threshold}");
            }

            #[cfg(feature = "sparse")]
            {
                use crate::solver::sparse::FaerSparseSolver;
                // Banded path: bandwidth declared, sparse threshold 0.
                let be = BackwardEulerSolver::new()
                    .with_sparse_solver(Box::new(FaerSparseSolver))
                    .with_jacobian_bandwidth(1)
                    .with_sparse_threshold(0)
                    .with_parallel_threshold(threshold);
                let cn = CrankNicolsonSolver::new()
                    .with_sparse_solver(Box::new(FaerSparseSolver))
                    .with_jacobian_bandwidth(1)
                    .with_sparse_threshold(0)
                    .with_parallel_threshold(threshold);
                let cases: [(&str, &dyn SteppableSolver); 2] = [
                    ("banded Backward Euler", &be),
                    ("banded Crank-Nicolson", &cn),
                ];
                for (name, solver) in cases {
                    let on_worker = evaluated_on_a_worker(n, |domain, chain, state| {
                        solver.step(domain, chain, state, &[], 0.0, 0.1)
                    });
                    assert_eq!(on_worker, parallel, "{name}, threshold {threshold}");
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn implicit_solvers_expose_their_jacobian_dispatch_threshold() {
        use crate::solver::methods::backward_euler::BackwardEulerSolver;
        use crate::solver::methods::bdf2::BDF2Solver;
        use crate::solver::methods::crank_nicolson::CrankNicolsonSolver;

        fn check<S>(
            new: fn() -> S,
            with: fn(S, usize) -> S,
            set: fn(&S, usize),
            get: fn(&S) -> usize,
        ) {
            assert_eq!(get(&new()), default_parallel_threshold());
            let solver = with(new(), 7);
            assert_eq!(get(&solver), 7);
            set(&solver, 9);
            assert_eq!(get(&solver), 9);
        }
        check(
            BackwardEulerSolver::new,
            BackwardEulerSolver::with_parallel_threshold,
            BackwardEulerSolver::set_parallel_threshold,
            BackwardEulerSolver::parallel_threshold,
        );
        check(
            CrankNicolsonSolver::new,
            CrankNicolsonSolver::with_parallel_threshold,
            CrankNicolsonSolver::set_parallel_threshold,
            CrankNicolsonSolver::parallel_threshold,
        );
        check(
            BDF2Solver::new,
            BDF2Solver::with_parallel_threshold,
            BDF2Solver::set_parallel_threshold,
            BDF2Solver::parallel_threshold,
        );
    }

    #[test]
    fn jacobian_of_linear_decay_is_minus_lambda_identity() {
        let lambda = 2.5;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(3));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let state = domain.model.initial_state(domain.mesh.as_ref());
        let jac = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.1,
            &JacobianDispatch::default(),
        )
        .unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { -lambda } else { 0.0 };
                assert!(
                    (jac[(i, j)] - expected).abs() < 1e-4,
                    "jac[{i},{j}] = {} (expected {expected})",
                    jac[(i, j)]
                );
            }
        }
    }

    #[test]
    fn backward_euler_theta_one_matches_analytical_for_linear_decay() {
        // For du/dt = -lambda*u, backward Euler gives:
        // u^{n+1} = u^n / (1 + lambda*dt)
        let lambda = 3.0;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = implicit_step_newton(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            ImplicitFormula::Theta(1.0),
            &NalgebraDenseSolver,
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let expected = 1.0 / (1.0 + lambda * dt);
        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn crank_nicolson_theta_half_matches_analytical_for_linear_decay() {
        // For du/dt = -lambda*u, Crank-Nicolson gives:
        // u^{n+1} = u^n * (1 - lambda*dt/2) / (1 + lambda*dt/2)
        let lambda = 3.0;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = implicit_step_newton(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            ImplicitFormula::Theta(0.5),
            &NalgebraDenseSolver,
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let expected = (1.0 - lambda * dt / 2.0) / (1.0 + lambda * dt / 2.0);
        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn backward_euler_stable_for_very_stiff_problem() {
        // lambda*dt = 1000 -- far beyond any explicit method's stability
        // limit. Backward Euler must remain bounded and well-behaved.
        let lambda = 1.0e4;
        let dt = 0.1; // lambda*dt = 1000
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = implicit_step_newton(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            ImplicitFormula::Theta(1.0),
            &NalgebraDenseSolver,
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!(v.is_finite(), "value diverged: {v}");
            assert!(v.abs() < 1.0, "expected strong damping, got {v}");
        }
    }

    // ── Dirichlet BC × perturbed Jacobian (#113) ───────────────────────────────

    use crate::boundary::PinnedNodes;

    /// Discrete tridiagonal diffusion with a spike initial condition —
    /// genuine spatial coupling to neighbours, needed to distinguish
    /// "legitimate physics coupling" from "spurious coupling introduced by
    /// BC re-application" in the test below. A local copy rather than
    /// reusing `sparse_tests::TridiagonalDiffusion`: that fixture only
    /// exists under `feature = "sparse"`, and this test needs no sparse
    /// machinery.
    #[derive(Debug)]
    struct TridiagonalDiffusion {
        diffusion: f64,
    }

    impl RequiresContext for TridiagonalDiffusion {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for TridiagonalDiffusion {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            let n = u.len();
            let d = self.diffusion;
            let out = DVector::from_fn(n, |i, _| {
                let left = if i > 0 { u[i - 1] } else { 0.0 };
                let right = if i + 1 < n { u[i + 1] } else { 0.0 };
                d * (left - 2.0 * u[i] + right)
            });
            Ok(ContextValue::ScalarField(out))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            let n = mesh.n_dof();
            ContextValue::ScalarField(DVector::from_fn(
                n,
                |i, _| if i == n / 2 { 1.0 } else { 0.0 },
            ))
        }

        fn name(&self) -> &str {
            "tridiagonal_diffusion_dirichlet_test"
        }
    }

    #[test]
    fn dirichlet_constrained_column_is_exactly_zero_row_keeps_real_coupling() {
        // Undeclared constrained node (no `constrained_nodes`, behaviour
        // before DD-070), kept as such. DD-033's known limitation, closed by
        // #113: `finite_difference_jacobian`
        // perturbs each state component and re-evaluates `evaluate_derivative`,
        // which re-applies boundary conditions on every call. A
        // Dirichlet-constrained node's perturbation is therefore overwritten
        // before `compute_physics` ever sees it -- physically correct (a
        // BC-constrained node isn't a free unknown), verified here rather
        // than merely documented.
        let n = 5;
        let constrained = n / 2; // the model's initial spike location
        let diffusion = 0.7;

        let domain = Domain::new(
            "dirichlet_test",
            Box::new(TridiagonalDiffusion { diffusion }),
            make_mesh(n),
        )
        .with_boundary_conditions(vec![Box::new(PinnedNodes::new(&[constrained], |_| 0.0))]);
        let scenario = Scenario::multi(vec![domain]).unwrap();
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let state = domain.model.initial_state(domain.mesh.as_ref());
        let jac = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.01,
            &JacobianDispatch::default(),
        )
        .unwrap();

        // Column: perturbing the constrained node itself must leave every
        // row's output exactly unchanged -- the perturbation never
        // survives BC re-application, so both the base and perturbed
        // evaluations see bit-identical states, giving f_j - f0 = 0
        // exactly, not merely small.
        for i in 0..n {
            assert_eq!(
                jac[(i, constrained)],
                0.0,
                "column {constrained} (the constrained node) must be exactly \
                 zero at row {i} -- got {}",
                jac[(i, constrained)]
            );
        }

        // Row: the constrained node's own output still genuinely depends
        // on its unconstrained neighbours through the model's real spatial
        // coupling -- legitimate physics, not spurious coupling, and must
        // not be zeroed out by the fix the column check above verifies.
        let left = constrained - 1;
        let right = constrained + 1;
        assert!(
            (jac[(constrained, left)] - diffusion).abs() < 1e-6,
            "expected left-neighbour coupling ~{diffusion}, got {}",
            jac[(constrained, left)]
        );
        assert!(
            (jac[(constrained, right)] - diffusion).abs() < 1e-6,
            "expected right-neighbour coupling ~{diffusion}, got {}",
            jac[(constrained, right)]
        );
    }

    #[test]
    fn declared_constrained_node_has_a_zero_row_and_column() {
        // DD-070 (#200): a declared node gets a zero derivative at every
        // evaluation, so its row of the Jacobian is zero as well as its
        // column -- an identity row in `I - theta*dt*J`. The other entries
        // are those of the undeclared case, bit for bit.
        let n = 5;
        let constrained = n / 2;
        let jacobian = |pin: PinnedNodes| {
            let domain = Domain::new(
                "dirichlet_test",
                Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
                make_mesh(n),
            )
            .with_boundary_conditions(vec![Box::new(pin)]);
            let state = domain.model.initial_state(domain.mesh.as_ref());
            let chain = build_calculator_chain(&domain.model.required_variables(), &[]).unwrap();
            finite_difference_jacobian(
                &domain,
                &chain,
                &state,
                0.0,
                0.01,
                &JacobianDispatch::default(),
            )
            .unwrap()
        };
        let undeclared = jacobian(PinnedNodes::new(&[constrained], |_| 0.0));
        let declared = jacobian(PinnedNodes::new(&[constrained], |_| 0.0).declared());

        for k in 0..n {
            assert_eq!(declared[(constrained, k)], 0.0, "row entry {k}");
            assert_eq!(declared[(k, constrained)], 0.0, "column entry {k}");
        }
        for i in (0..n).filter(|&i| i != constrained) {
            for j in 0..n {
                assert_eq!(
                    declared[(i, j)].to_bits(),
                    undeclared[(i, j)].to_bits(),
                    "({i}, {j})"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "sparse"))]
mod sparse_tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::chain::build_calculator_chain;
    use crate::solver::linear::NalgebraDenseSolver;
    use crate::solver::scenario::Scenario;
    use crate::solver::sparse::FaerSparseSolver;

    /// Discrete tridiagonal diffusion — `du_i/dt = D*(u[i-1] - 2u[i] +
    /// u[i+1])`, zero outside the domain. Not a real `BoundaryCondition` —
    /// this fixture exists only to exercise the banded path with a
    /// genuinely local, bandwidth-1 coupling (real Dirichlet/Neumann
    /// boundary handling is covered separately, see the module's own docs
    /// on boundary conditions × the perturbed Jacobian, and
    /// `tests::dirichlet_constrained_column_is_exactly_zero_row_keeps_real_coupling`,
    /// #113).
    #[derive(Debug)]
    struct TridiagonalDiffusion {
        diffusion: f64,
    }

    impl RequiresContext for TridiagonalDiffusion {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for TridiagonalDiffusion {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            let n = u.len();
            let d = self.diffusion;
            let out = DVector::from_fn(n, |i, _| {
                let left = if i > 0 { u[i - 1] } else { 0.0 };
                let right = if i + 1 < n { u[i + 1] } else { 0.0 };
                d * (left - 2.0 * u[i] + right)
            });
            Ok(ContextValue::ScalarField(out))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            let n = mesh.n_dof();
            ContextValue::ScalarField(DVector::from_fn(
                n,
                |i, _| if i == n / 2 { 1.0 } else { 0.0 },
            ))
        }

        fn name(&self) -> &str {
            "tridiagonal_diffusion_test"
        }
    }

    /// A [`SparseLinearSolver`] that panics if ever called — used to prove
    /// the dense path was actually taken (threshold/bandwidth conditions
    /// not met), not just that the result happens to look dense-like.
    struct PanicIfCalled;

    impl SparseLinearSolver for PanicIfCalled {
        fn solve(
            &self,
            _a: &faer::sparse::SparseColMat<usize, f64>,
            _b: &DVector<f64>,
        ) -> Result<DVector<f64>, OxiflowError> {
            panic!("sparse solver called when the dense path should have been taken");
        }
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    /// Builds the dense Jacobian and the banded one of `scenario` at its
    /// initial state, and checks that they are equal bit for bit.
    ///
    /// Inside the band, a row of a color evaluation reads exactly the
    /// unknowns it reads in the dense evaluation of that column (the other
    /// perturbed columns of the color are outside its stencil), so every
    /// entry is computed from the same operands with the same operations.
    /// Outside the band the dense entries are exactly zero for the same
    /// reason, and the banded matrix stores nothing.
    fn assert_banded_jacobian_equals_dense(scenario: &Scenario, bandwidth: usize) {
        let domain = scenario.single_domain().unwrap();
        let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
        let state = domain.model.initial_state(domain.mesh.as_ref());

        let dense = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.01,
            &JacobianDispatch::default(),
        )
        .unwrap();
        let banded = banded_finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.01,
            bandwidth,
            &JacobianDispatch::default(),
        )
        .unwrap();
        let n = dense.nrows();
        assert_eq!((banded.nrows(), banded.ncols()), (n, n));

        let mut stored = 0;
        for entry in banded.as_ref().triplet_iter() {
            let (i, j) = (entry.row, entry.col);
            assert!(i.abs_diff(j) <= bandwidth, "({i},{j}) outside the band");
            assert_eq!(
                entry.val.to_bits(),
                dense[(i, j)].to_bits(),
                "({i},{j}): banded {} vs dense {}",
                entry.val,
                dense[(i, j)]
            );
            stored += 1;
        }
        let band_size: usize = (0..n)
            .map(|j| (j + bandwidth + 1).min(n) - j.saturating_sub(bandwidth))
            .sum();
        assert_eq!(stored, band_size);

        for i in 0..n {
            for j in 0..n {
                if i.abs_diff(j) > bandwidth {
                    assert_eq!(dense[(i, j)], 0.0, "dense ({i},{j}) outside the band");
                }
            }
        }
    }

    #[test]
    fn banded_jacobian_equals_dense_jacobian_on_a_scalar_field() {
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
            make_mesh(12),
        );
        assert_banded_jacobian_equals_dense(&scenario, 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn banded_jacobian_is_bit_identical_in_sequential_and_parallel_dispatch() {
        use super::tests::{coupled_reaction_diffusion, dispatch_with_threshold};

        let cases = [
            (
                Scenario::single(
                    Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
                    make_mesh(12),
                ),
                1,
            ),
            (
                Scenario::single(Box::new(coupled_reaction_diffusion()), make_mesh(6)),
                3,
            ),
        ];
        for (scenario, bandwidth) in &cases {
            let domain = scenario.single_domain().unwrap();
            let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
            let state = domain.model.initial_state(domain.mesh.as_ref());
            let jacobian = |threshold| {
                let dispatch = dispatch_with_threshold(threshold);
                banded_finite_difference_jacobian(
                    domain, &chain, &state, 0.0, 0.01, *bandwidth, &dispatch,
                )
                .unwrap()
            };
            let (parallel, sequential) = (jacobian(1), jacobian(usize::MAX));
            assert_eq!(
                parallel.symbolic().col_ptr(),
                sequential.symbolic().col_ptr()
            );
            assert_eq!(
                parallel.symbolic().row_idx(),
                sequential.symbolic().row_idx()
            );
            assert!(parallel
                .val()
                .iter()
                .zip(sequential.val())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn banded_jacobian_wider_than_the_system_is_the_full_dense_jacobian() {
        // Fewer columns than colors: every column gets its own color and
        // every band covers the whole column.
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
            make_mesh(4),
        );
        assert_banded_jacobian_equals_dense(&scenario, 6);
    }

    #[test]
    fn banded_jacobian_equals_dense_jacobian_on_a_vector_field() {
        let scenario = Scenario::single(
            Box::new(super::tests::coupled_reaction_diffusion()),
            make_mesh(6),
        );
        // d·(s + 1) − 1 with d = 2 components and a stencil of half-width 1.
        assert_banded_jacobian_equals_dense(&scenario, 3);
    }

    #[test]
    fn adaptive_step_with_no_bandwidth_matches_the_dense_step_exactly() {
        let n = 6;
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.5 }),
            make_mesh(n),
        );
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state_a = domain.model.initial_state(domain.mesh.as_ref());
        let mut state_b = state_a.clone();

        let via_adaptive = implicit_step_adaptive(
            domain,
            &chain,
            &mut state_a,
            0.0,
            0.05,
            ImplicitFormula::Theta(1.0),
            &NalgebraDenseSolver,
            &PanicIfCalled,
            100,
            None, // no bandwidth declared -> dense path, unconditionally
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let via_plain = implicit_step_newton(
            domain,
            &chain,
            &mut state_b,
            0.0,
            0.05,
            ImplicitFormula::Theta(1.0),
            &NalgebraDenseSolver,
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let a = via_adaptive.as_scalar_field().unwrap();
        let b = via_plain.as_scalar_field().unwrap();
        for i in 0..n {
            assert!(
                (a[i] - b[i]).abs() < 1e-12,
                "mismatch at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn small_system_stays_dense_even_with_bandwidth_declared() {
        let n = 6; // below sparse_threshold below
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.5 }),
            make_mesh(n),
        );
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();
        let mut state = domain.model.initial_state(domain.mesh.as_ref());

        // PanicIfCalled proves the sparse path was never entered: bandwidth
        // is declared (Some(1)) but n (6) does not exceed sparse_threshold
        // (100), so the dense path must be taken regardless.
        let result = implicit_step_adaptive(
            domain,
            &chain,
            &mut state,
            0.0,
            0.05,
            ImplicitFormula::Theta(1.0),
            &NalgebraDenseSolver,
            &PanicIfCalled,
            100,
            Some(1),
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        assert!(result
            .as_scalar_field()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn adaptive_step_with_bandwidth_matches_dense_result_above_threshold() {
        let n = 40; // above the small sparse_threshold used below
        let diffusion = 0.3;
        let dt = 0.02;

        let scenario_a =
            Scenario::single(Box::new(TridiagonalDiffusion { diffusion }), make_mesh(n));
        let domain_a = scenario_a.single_domain().unwrap();
        let requirements_a = scenario_a.context_requirements();
        let chain_a = build_calculator_chain(&requirements_a, &[]).unwrap();
        let mut state_a = domain_a.model.initial_state(domain_a.mesh.as_ref());

        let scenario_b =
            Scenario::single(Box::new(TridiagonalDiffusion { diffusion }), make_mesh(n));
        let domain_b = scenario_b.single_domain().unwrap();
        let requirements_b = scenario_b.context_requirements();
        let chain_b = build_calculator_chain(&requirements_b, &[]).unwrap();
        let mut state_b = domain_b.model.initial_state(domain_b.mesh.as_ref());

        let via_sparse = implicit_step_adaptive(
            domain_a,
            &chain_a,
            &mut state_a,
            0.0,
            dt,
            ImplicitFormula::Theta(1.0),
            &NalgebraDenseSolver,
            &FaerSparseSolver,
            10, // sparse_threshold well below n=40
            Some(1),
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let via_dense = implicit_step_newton(
            domain_b,
            &chain_b,
            &mut state_b,
            0.0,
            dt,
            ImplicitFormula::Theta(1.0),
            &NalgebraDenseSolver,
            &NewtonParams::default(),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let a = via_sparse.as_scalar_field().unwrap();
        let b = via_dense.as_scalar_field().unwrap();
        for i in 0..n {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "sparse/dense mismatch at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    // ── Iterated Newton on the banded path (#206) ────────────────────────────

    use crate::model::CubicDecay;

    /// One theta-method step of f(u) = -u^3 from u = 1 at `dt` = 0.5,
    /// through `implicit_step_adaptive` with a declared bandwidth of 1.
    fn cubic_decay_step(
        theta: f64,
        sparse_solver: &dyn SparseLinearSolver,
        sparse_threshold: usize,
        newton: &NewtonParams,
    ) -> Result<ContextValue, OxiflowError> {
        let scenario = Scenario::single(Box::new(CubicDecay), make_mesh(4));
        let domain = scenario.single_domain().unwrap();
        let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        implicit_step_adaptive(
            domain,
            &chain,
            &mut state,
            0.0,
            0.5,
            ImplicitFormula::Theta(theta),
            &NalgebraDenseSolver,
            sparse_solver,
            sparse_threshold,
            Some(1),
            newton,
            &JacobianDispatch::default(),
        )
    }

    /// Root of the theta-method step of f(u) = -u^3 from u = 1:
    /// u + theta*dt*u^3 = 1 - (1 - theta)*dt, solved by scalar Newton.
    fn cubic_decay_root(theta: f64, dt: f64) -> f64 {
        let rhs = 1.0 - (1.0 - theta) * dt;
        let mut u = 1.0_f64;
        for _ in 0..100 {
            let g = u + theta * dt * u.powi(3) - rhs;
            let dg = 1.0 + 3.0 * theta * dt * u.powi(2);
            u -= g / dg;
        }
        u
    }

    /// Converges far below the default tolerance, so that a step which
    /// stops early, or never iterates, misses the root.
    fn tight_full_newton() -> NewtonParams {
        NewtonParams {
            convergence: NewtonConvergence::ResidualOnly {
                tol_abs: 1e-13,
                tol_rel: 0.0,
            },
            jacobian_strategy: JacobianStrategy::FullNewton,
            max_iterations: 10,
        }
    }

    #[test]
    fn banded_path_iterates_newton_to_the_root() {
        for theta in [1.0, 0.5] {
            let expected = cubic_decay_root(theta, 0.5);
            let next = cubic_decay_step(theta, &FaerSparseSolver, 0, &tight_full_newton()).unwrap();
            for v in next.as_scalar_field().unwrap().iter() {
                assert!(
                    (v - expected).abs() < 1e-12,
                    "theta {theta}: got {v}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn banded_path_reports_an_unconverged_newton_loop() {
        // One frozen-Jacobian correction does not reach the default
        // tolerance at dt = 0.5: the banded path must say so, not return
        // the corrected state.
        for theta in [1.0, 0.5] {
            let err = cubic_decay_step(theta, &FaerSparseSolver, 0, &NewtonParams::default())
                .unwrap_err();
            assert!(
                matches!(err, OxiflowError::NewtonNotConverged { iterations: 1, .. }),
                "theta {theta}: got {err:?}"
            );
        }
    }

    #[test]
    fn dense_fallback_uses_the_configured_newton_parameters() {
        // A sparse solver is configured and a bandwidth declared, but the
        // system (4 unknowns) does not exceed the threshold: the dense path
        // runs, under the caller's parameters.
        for theta in [1.0, 0.5] {
            let short_budget = NewtonParams {
                max_iterations: 2,
                ..NewtonParams::default()
            };
            let err = cubic_decay_step(theta, &PanicIfCalled, 100, &short_budget).unwrap_err();
            assert!(
                matches!(err, OxiflowError::NewtonNotConverged { iterations: 2, .. }),
                "theta {theta}: got {err:?}"
            );

            let expected = cubic_decay_root(theta, 0.5);
            let next = cubic_decay_step(theta, &PanicIfCalled, 100, &tight_full_newton()).unwrap();
            for v in next.as_scalar_field().unwrap().iter() {
                assert!(
                    (v - expected).abs() < 1e-12,
                    "theta {theta}: got {v}, expected {expected}"
                );
            }
        }
    }
}
