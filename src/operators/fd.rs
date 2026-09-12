//! # Module `operators::fd`
//!
//! Finite-difference [`DiscreteOperator`] implementations for
//! [`UniformGrid1D`] — `UpwindGradient` (1st order, advection-dominant),
//! `CenteredGradient` (2nd order, diffusion-dominant), `CenteredLaplacian`
//! (2nd order) (#47, DD-012).
//!
//! ## `Direction` rather than two separate structs
//!
//! `Forward` and `Backward` differences share the exact same boundary
//! fallback (each is the other's substitute at the far boundary), so a
//! single `UpwindGradient` parameterized by [`Direction`] avoids duplicating
//! that fallback logic in two otherwise-identical structs — validated in
//! this sprint's design review before implementation.
//!
//! ## `*_from_dx` associated functions — why the stencil math is mesh-free
//!
//! [`crate::context::calculators::spatial::FDGradientCalculator`] and
//! `FDLaplacianCalculator` (the FD delegation refactor landing alongside
//! `#47`) hold `Arc<dyn Mesh>` (INV-1, object-safe interface) — not a
//! concrete mesh type. [`DiscreteOperator::apply`] requires `&Self::MeshType`
//! (a concrete type, DD-012), which a `dyn Mesh` cannot produce without
//! downcasting. The only mesh datum any FD stencil actually needs is the
//! single scalar `dx = mesh.characteristic_length()` — already part of the
//! object-safe `Mesh` trait. The stencil math is therefore factored into
//! `pub(crate)` `compute_from_dx(..., dx: f64, ...)` functions, called from
//! both `apply()` (concrete mesh, e.g. a future generic pipeline consumer)
//! and directly from the calculators (`Arc<dyn Mesh>` path) — one
//! implementation of the math either way, no change needed to the `Mesh`
//! trait itself, no breaking change to the calculators' existing
//! `Arc<dyn Mesh>`-based public API.
//!
//! [`UniformGrid1D`]: crate::mesh::structured::UniformGrid1D
//!
//! ## Rayon dispatch (`parallel` feature, DD-014, #51)
//!
//! Each `compute_from_dx` takes an explicit `parallel_threshold: usize`. Past
//! that node count, the per-index stencil closure runs through Rayon's
//! `IntoParallelIterator` instead of a plain `for` loop; below it, the
//! sequential path always runs, `parallel` feature or not. The
//! threshold is a parameter rather than a field on these operator structs —
//! the actual value is owned by the calculators that call this function
//! directly (see [`crate::context::calculators::spatial::FDGradientCalculator`]
//! and `FDLaplacianCalculator`'s `parallel_threshold` field), mirroring how
//! `SparseLinearSolver`'s threshold (DD-043, #103) lives on the caller rather
//! than the callee. [`DiscreteOperator::apply`] has no such caller to draw a
//! value from, so it uses [`default_parallel_threshold`], calibrated
//! empirically at 49_999: sequential cost of a single stencil evaluation
//! here (2-3 flops) is ~1 ns/node, the same order of magnitude as typical
//! Rayon dispatch overhead well below that count. This does not transfer
//! from `chrom-rs`'s threshold of 999, which was calibrated for a much
//! heavier per-node cost (e.g. an isotherm evaluation) — only the
//! operation-volume semantics carries over, not the number itself.

use nalgebra::DVector;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::mesh::structured::UniformGrid1D;
use crate::mesh::Mesh;
use crate::operators::DiscreteOperator;

/// Default Rayon dispatch threshold for the `compute_from_dx` functions in
/// this module (DD-014, #51) — see the module documentation for the
/// empirical justification. Used by [`DiscreteOperator::apply`], which has
/// no per-calculator config to draw a threshold from; calculators
/// (`FDGradientCalculator`/`FDLaplacianCalculator`) use this as their own
/// field default but may override it via `with_parallel_threshold`. Gated
/// like everything else here: every call site already only exists under
/// `parallel`, so an unconditional definition would sit unused (and warn)
/// in a default build.
#[cfg(feature = "parallel")]
pub(crate) fn default_parallel_threshold() -> usize {
    49_999
}

// ── Direction ───────────────────────────────────────────────────────────────

/// Direction of a 1st-order upwind (decentered) stencil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `(u[i+1] − u[i]) / dx`; falls back to `Backward` at the last node.
    Forward,
    /// `(u[i] − u[i−1]) / dx`; falls back to `Forward` at the first node.
    Backward,
}

// ── UpwindGradient ────────────────────────────────────────────────────────────

/// 1st-order upwind (decentered) gradient — advection-dominant schemes.
///
/// See the module documentation for why a single type parameterized by
/// [`Direction`] replaces two separate `Forward`/`Backward` structs.
#[derive(Debug, Clone, Copy)]
pub struct UpwindGradient {
    direction: Direction,
}

impl UpwindGradient {
    /// Creates an upwind gradient operator biased in `direction`.
    pub fn new(direction: Direction) -> Self {
        Self { direction }
    }

    /// Stencil math on a raw field and scalar `dx` — see the module
    /// documentation for why this bypasses `Self::MeshType`, and for the
    /// `parallel_threshold` dispatch (DD-014, #51).
    ///
    /// Two versions exist, split by `#[cfg(feature = "parallel")]` rather
    /// than one version that takes and ignores the threshold: a discarded
    /// parameter still gets computed and passed at every call site for no
    /// effect, which is confusing to read and contradicts the "genuinely
    /// unrepresentable without the feature" posture already taken for the
    /// calculators' own field (see `context::calculators::spatial`).
    #[cfg(feature = "parallel")]
    pub(crate) fn compute_from_dx(
        u: &DVector<f64>,
        dx: f64,
        direction: Direction,
        parallel_threshold: usize,
    ) -> Result<DVector<f64>, OxiflowError> {
        let n = u.len();
        if n < 2 {
            return Err(OxiflowError::InvalidDomain(format!(
                "UpwindGradient requires at least 2 nodes, got {n}"
            )));
        }

        let stencil = |i: usize| -> f64 {
            match direction {
                Direction::Forward => {
                    if i < n - 1 {
                        (u[i + 1] - u[i]) / dx
                    } else {
                        // Right boundary fallback: backward.
                        (u[n - 1] - u[n - 2]) / dx
                    }
                }
                Direction::Backward => {
                    if i > 0 {
                        (u[i] - u[i - 1]) / dx
                    } else {
                        // Left boundary fallback: forward.
                        (u[1] - u[0]) / dx
                    }
                }
            }
        };

        if n >= parallel_threshold {
            let values: Vec<f64> = (0..n).into_par_iter().map(stencil).collect();
            return Ok(DVector::from_vec(values));
        }

        let mut grad = DVector::zeros(n);
        for i in 0..n {
            grad[i] = stencil(i);
        }
        Ok(grad)
    }

    /// Sequential-only counterpart of the function above, compiled when the
    /// `parallel` feature is off — no threshold parameter, because there is
    /// nothing to threshold against without Rayon in the dependency tree.
    #[cfg(not(feature = "parallel"))]
    pub(crate) fn compute_from_dx(
        u: &DVector<f64>,
        dx: f64,
        direction: Direction,
    ) -> Result<DVector<f64>, OxiflowError> {
        let n = u.len();
        if n < 2 {
            return Err(OxiflowError::InvalidDomain(format!(
                "UpwindGradient requires at least 2 nodes, got {n}"
            )));
        }

        let mut grad = DVector::zeros(n);
        for i in 0..n {
            grad[i] = match direction {
                Direction::Forward => {
                    if i < n - 1 {
                        (u[i + 1] - u[i]) / dx
                    } else {
                        (u[n - 1] - u[n - 2]) / dx
                    }
                }
                Direction::Backward => {
                    if i > 0 {
                        (u[i] - u[i - 1]) / dx
                    } else {
                        (u[1] - u[0]) / dx
                    }
                }
            };
        }
        Ok(grad)
    }
}

impl DiscreteOperator for UpwindGradient {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
    ) -> Result<ContextValue, OxiflowError> {
        let u = field.as_scalar_field()?;
        let dx = mesh.characteristic_length();
        #[cfg(feature = "parallel")]
        let grad = Self::compute_from_dx(u, dx, self.direction, default_parallel_threshold())?;
        #[cfg(not(feature = "parallel"))]
        let grad = Self::compute_from_dx(u, dx, self.direction)?;
        Ok(ContextValue::ScalarField(grad))
    }
}

// ── CenteredGradient ──────────────────────────────────────────────────────────

/// 2nd-order centered gradient — diffusion-dominant schemes.
///
/// Boundary nodes fall back to a 1st-order one-sided stencil (same posture
/// as `FDScheme::Central` prior to this refactor).
#[derive(Debug, Clone, Copy, Default)]
pub struct CenteredGradient;

impl CenteredGradient {
    /// Creates a centered gradient operator.
    pub fn new() -> Self {
        Self
    }

    /// Stencil math on a raw field and scalar `dx` — see the module
    /// documentation for why this bypasses `Self::MeshType`, and for why
    /// `parallel_threshold` (DD-014, #51) is split across two cfg-gated
    /// versions rather than a single one that sometimes ignores it.
    #[cfg(feature = "parallel")]
    pub(crate) fn compute_from_dx(
        u: &DVector<f64>,
        dx: f64,
        parallel_threshold: usize,
    ) -> Result<DVector<f64>, OxiflowError> {
        let n = u.len();
        if n < 2 {
            return Err(OxiflowError::InvalidDomain(format!(
                "CenteredGradient requires at least 2 nodes, got {n}"
            )));
        }

        let stencil = |i: usize| -> f64 {
            if i == 0 {
                // Left boundary: 1st-order forward.
                (u[1] - u[0]) / dx
            } else if i == n - 1 {
                // Right boundary: 1st-order backward.
                (u[n - 1] - u[n - 2]) / dx
            } else {
                // Interior: 2nd-order central.
                (u[i + 1] - u[i - 1]) / (2.0 * dx)
            }
        };

        if n >= parallel_threshold {
            let values: Vec<f64> = (0..n).into_par_iter().map(stencil).collect();
            return Ok(DVector::from_vec(values));
        }

        let mut grad = DVector::zeros(n);
        for i in 0..n {
            grad[i] = stencil(i);
        }
        Ok(grad)
    }

    /// Sequential-only counterpart, compiled when the `parallel` feature is
    /// off — see [`UpwindGradient::compute_from_dx`]'s equivalent split.
    #[cfg(not(feature = "parallel"))]
    pub(crate) fn compute_from_dx(u: &DVector<f64>, dx: f64) -> Result<DVector<f64>, OxiflowError> {
        let n = u.len();
        if n < 2 {
            return Err(OxiflowError::InvalidDomain(format!(
                "CenteredGradient requires at least 2 nodes, got {n}"
            )));
        }

        let mut grad = DVector::zeros(n);
        for i in 0..n {
            grad[i] = if i == 0 {
                (u[1] - u[0]) / dx
            } else if i == n - 1 {
                (u[n - 1] - u[n - 2]) / dx
            } else {
                (u[i + 1] - u[i - 1]) / (2.0 * dx)
            };
        }
        Ok(grad)
    }
}

impl DiscreteOperator for CenteredGradient {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
    ) -> Result<ContextValue, OxiflowError> {
        let u = field.as_scalar_field()?;
        let dx = mesh.characteristic_length();
        #[cfg(feature = "parallel")]
        let grad = Self::compute_from_dx(u, dx, default_parallel_threshold())?;
        #[cfg(not(feature = "parallel"))]
        let grad = Self::compute_from_dx(u, dx)?;
        Ok(ContextValue::ScalarField(grad))
    }
}

// ── CenteredLaplacian ─────────────────────────────────────────────────────────

/// 2nd-order centered Laplacian `∇²u = d²u/dx²`.
///
/// Boundary nodes use a 1st-order one-sided 3-point stencil (same posture as
/// `FDLaplacianCalculator` prior to this refactor).
#[derive(Debug, Clone, Copy, Default)]
pub struct CenteredLaplacian;

impl CenteredLaplacian {
    /// Creates a centered Laplacian operator.
    pub fn new() -> Self {
        Self
    }

    /// Stencil math on a raw field and scalar `dx` — see the module
    /// documentation for why this bypasses `Self::MeshType`, and for why
    /// `parallel_threshold` (DD-014, #51) is split across two cfg-gated
    /// versions rather than a single one that sometimes ignores it.
    ///
    /// Only the interior loop is dispatched through Rayon: the two boundary
    /// nodes are O(1) work each and not worth the dispatch overhead at any
    /// threshold this module would ever pick.
    #[cfg(feature = "parallel")]
    pub(crate) fn compute_from_dx(
        u: &DVector<f64>,
        dx: f64,
        parallel_threshold: usize,
    ) -> Result<DVector<f64>, OxiflowError> {
        let n = u.len();
        if n < 3 {
            return Err(OxiflowError::InvalidDomain(format!(
                "CenteredLaplacian requires at least 3 nodes, got {n}"
            )));
        }

        let dx2 = dx * dx;
        let mut lap = DVector::zeros(n);

        // Left boundary: one-sided stencil using nodes [0, 1, 2].
        lap[0] = (u[0] - 2.0 * u[1] + u[2]) / dx2;
        // Right boundary: one-sided stencil using nodes [n-3, n-2, n-1].
        lap[n - 1] = (u[n - 3] - 2.0 * u[n - 2] + u[n - 1]) / dx2;

        let interior_stencil = |i: usize| -> f64 { (u[i - 1] - 2.0 * u[i] + u[i + 1]) / dx2 };

        if n >= parallel_threshold {
            let values: Vec<f64> = (1..n - 1).into_par_iter().map(interior_stencil).collect();
            for (offset, v) in values.into_iter().enumerate() {
                lap[1 + offset] = v;
            }
            return Ok(lap);
        }

        // Interior: standard 3-point central difference.
        for i in 1..n - 1 {
            lap[i] = interior_stencil(i);
        }

        Ok(lap)
    }

    /// Sequential-only counterpart, compiled when the `parallel` feature is
    /// off — see [`UpwindGradient::compute_from_dx`]'s equivalent split.
    #[cfg(not(feature = "parallel"))]
    pub(crate) fn compute_from_dx(u: &DVector<f64>, dx: f64) -> Result<DVector<f64>, OxiflowError> {
        let n = u.len();
        if n < 3 {
            return Err(OxiflowError::InvalidDomain(format!(
                "CenteredLaplacian requires at least 3 nodes, got {n}"
            )));
        }

        let dx2 = dx * dx;
        let mut lap = DVector::zeros(n);

        lap[0] = (u[0] - 2.0 * u[1] + u[2]) / dx2;
        lap[n - 1] = (u[n - 3] - 2.0 * u[n - 2] + u[n - 1]) / dx2;

        for i in 1..n - 1 {
            lap[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / dx2;
        }

        Ok(lap)
    }
}

impl DiscreteOperator for CenteredLaplacian {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
    ) -> Result<ContextValue, OxiflowError> {
        let u = field.as_scalar_field()?;
        let dx = mesh.characteristic_length();
        #[cfg(feature = "parallel")]
        let lap = Self::compute_from_dx(u, dx, default_parallel_threshold())?;
        #[cfg(not(feature = "parallel"))]
        let lap = Self::compute_from_dx(u, dx)?;
        Ok(ContextValue::ScalarField(lap))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A threshold no test field ever reaches — forces the sequential path
    /// for tests whose intent is stencil correctness, not dispatch. Only
    /// meaningful with the `parallel` feature, hence the helpers below
    /// rather than passing this at every call site.
    #[cfg(feature = "parallel")]
    const SEQ: usize = usize::MAX;
    /// A threshold every non-empty test field reaches — forces the Rayon
    /// path where the `parallel` feature is enabled.
    #[cfg(feature = "parallel")]
    const PAR: usize = 0;

    // `compute_from_dx`'s arity itself depends on `parallel` (DD-014, #51) —
    // these wrappers absorb that so every correctness test below can call a
    // single, uniform signature regardless of feature. `PAR`/`SEQ`-driven
    // dispatch is exercised separately, only under `parallel`, further down.

    fn upwind_seq(
        u: &DVector<f64>,
        dx: f64,
        direction: Direction,
    ) -> Result<DVector<f64>, OxiflowError> {
        #[cfg(feature = "parallel")]
        {
            UpwindGradient::compute_from_dx(u, dx, direction, SEQ)
        }
        #[cfg(not(feature = "parallel"))]
        {
            UpwindGradient::compute_from_dx(u, dx, direction)
        }
    }

    fn centered_gradient_seq(u: &DVector<f64>, dx: f64) -> Result<DVector<f64>, OxiflowError> {
        #[cfg(feature = "parallel")]
        {
            CenteredGradient::compute_from_dx(u, dx, SEQ)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CenteredGradient::compute_from_dx(u, dx)
        }
    }

    fn centered_laplacian_seq(u: &DVector<f64>, dx: f64) -> Result<DVector<f64>, OxiflowError> {
        #[cfg(feature = "parallel")]
        {
            CenteredLaplacian::compute_from_dx(u, dx, SEQ)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CenteredLaplacian::compute_from_dx(u, dx)
        }
    }

    fn mesh(n: usize) -> UniformGrid1D {
        UniformGrid1D::new(n, 0.0, 1.0).unwrap()
    }

    /// Max absolute error over the given node indices.
    fn max_error(computed: &DVector<f64>, analytical: &DVector<f64>, indices: &[usize]) -> f64 {
        indices
            .iter()
            .map(|&i| (computed[i] - analytical[i]).abs())
            .fold(0.0, f64::max)
    }

    fn interior(n: usize) -> Vec<usize> {
        (1..n - 1).collect()
    }

    fn all_nodes(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    // ── UpwindGradient — analytical check ─────────────────────────────────────

    #[test]
    fn upwind_forward_on_linear_field_is_exact() {
        // u = x  →  ∂u/∂x = 1 everywhere (forward difference is exact on a
        // linear field, order verification below uses a non-linear field
        // instead, where the 1st-order error term is actually non-zero).
        let m = mesh(5);
        let dx = m.characteristic_length();
        let u = DVector::from_vec((0..5).map(|i| i as f64 * dx).collect());
        let grad = upwind_seq(&u, dx, Direction::Forward).unwrap();
        assert!(grad.iter().all(|&g| (g - 1.0).abs() < 1e-10));
    }

    #[test]
    fn upwind_backward_fallback_at_left_boundary() {
        let m = mesh(5);
        let dx = m.characteristic_length();
        let u = DVector::from_vec((0..5).map(|i| i as f64 * dx).collect());
        let grad = upwind_seq(&u, dx, Direction::Backward).unwrap();
        assert!((grad[0] - 1.0).abs() < 1e-10);
    }

    // ── UpwindGradient — order verification (grid refinement) ────────────────

    #[test]
    fn upwind_forward_is_first_order() {
        // u = x²  →  ∂u/∂x = 2x. Forward difference: O(dx) everywhere.
        let errors: Vec<f64> = [21usize, 41]
            .iter()
            .map(|&n| {
                let m = mesh(n);
                let dx = m.characteristic_length();
                let u = DVector::from_vec((0..n).map(|i| (i as f64 * dx).powi(2)).collect());
                let analytical = DVector::from_vec((0..n).map(|i| 2.0 * i as f64 * dx).collect());
                let grad = upwind_seq(&u, dx, Direction::Forward).unwrap();
                max_error(&grad, &analytical, &all_nodes(n))
            })
            .collect();

        let ratio = errors[0] / errors[1];
        // h → h/2 should roughly halve the error for a 1st-order scheme.
        assert!(
            (1.5..=2.5).contains(&ratio),
            "expected ratio ≈ 2, got {ratio}"
        );
    }

    // ── CenteredGradient — analytical check ───────────────────────────────────

    #[test]
    fn centered_gradient_of_linear_field_is_exact() {
        let m = mesh(5);
        let dx = m.characteristic_length();
        let u = DVector::from_vec((0..5).map(|i| i as f64 * dx).collect());
        let grad = centered_gradient_seq(&u, dx).unwrap();
        assert!(grad.iter().all(|&g| (g - 1.0).abs() < 1e-10));
    }

    // ── CenteredGradient — order verification (grid refinement) ──────────────

    #[test]
    fn centered_gradient_is_second_order_at_interior_nodes() {
        // u = sin(x)  →  ∂u/∂x = cos(x). Centered difference: O(dx²) at
        // interior nodes (boundary nodes stay 1st order, excluded here).
        let errors: Vec<f64> = [21usize, 41]
            .iter()
            .map(|&n| {
                let m = mesh(n);
                let dx = m.characteristic_length();
                let u = DVector::from_vec((0..n).map(|i| (i as f64 * dx).sin()).collect());
                let analytical = DVector::from_vec((0..n).map(|i| (i as f64 * dx).cos()).collect());
                let grad = centered_gradient_seq(&u, dx).unwrap();
                max_error(&grad, &analytical, &interior(n))
            })
            .collect();

        let ratio = errors[0] / errors[1];
        // h → h/2 should roughly quarter the error for a 2nd-order scheme.
        assert!(
            (3.0..=5.0).contains(&ratio),
            "expected ratio ≈ 4, got {ratio}"
        );
    }

    // ── CenteredLaplacian — analytical check ──────────────────────────────────

    #[test]
    fn centered_laplacian_of_quadratic_field_is_exact_at_interior() {
        // u = x²  →  ∇²u = 2 everywhere (exact for the 3-point stencil).
        let m = mesh(7);
        let dx = m.characteristic_length();
        let n = 7;
        let u = DVector::from_vec((0..n).map(|i| (i as f64 * dx).powi(2)).collect());
        let lap = centered_laplacian_seq(&u, dx).unwrap();
        for &i in &interior(n) {
            assert!((lap[i] - 2.0).abs() < 1e-8, "node {i}: got {}", lap[i]);
        }
    }

    // ── CenteredLaplacian — order verification (grid refinement) ─────────────

    #[test]
    fn centered_laplacian_is_second_order_at_interior_nodes() {
        // u = sin(x)  →  ∇²u = -sin(x). 3-point centered: O(dx²) at interior
        // nodes (boundary stencil stays 1st order, excluded here).
        let errors: Vec<f64> = [21usize, 41]
            .iter()
            .map(|&n| {
                let m = mesh(n);
                let dx = m.characteristic_length();
                let u = DVector::from_vec((0..n).map(|i| (i as f64 * dx).sin()).collect());
                let analytical =
                    DVector::from_vec((0..n).map(|i| -(i as f64 * dx).sin()).collect());
                let lap = centered_laplacian_seq(&u, dx).unwrap();
                max_error(&lap, &analytical, &interior(n))
            })
            .collect();

        let ratio = errors[0] / errors[1];
        assert!(
            (3.0..=5.0).contains(&ratio),
            "expected ratio ≈ 4, got {ratio}"
        );
    }

    // ── InvalidDomain on out-of-bounds field size ─────────────────────────────

    #[test]
    fn upwind_gradient_rejects_single_node_field() {
        let err = upwind_seq(&DVector::from_vec(vec![1.0]), 0.1, Direction::Forward).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    #[test]
    fn centered_gradient_rejects_single_node_field() {
        let err = centered_gradient_seq(&DVector::from_vec(vec![1.0]), 0.1).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    #[test]
    fn centered_laplacian_rejects_two_node_field() {
        let err = centered_laplacian_seq(&DVector::from_vec(vec![0.0, 1.0]), 0.1).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    // ── Parallel/sequential equivalence (DD-014, #51) ─────────────────────────
    //
    // `PAR` (threshold 0) forces the Rayon path on any non-empty field —
    // gated behind `feature = "parallel"` since the branch does not exist
    // otherwise. Each test compares against the same field run through
    // `SEQ`, which always takes the plain loop regardless of feature.

    #[cfg(feature = "parallel")]
    #[test]
    fn upwind_gradient_parallel_matches_sequential() {
        let m = mesh(64);
        let dx = m.characteristic_length();
        let u = DVector::from_vec((0..64).map(|i| (i as f64 * dx).sin()).collect());
        let seq = UpwindGradient::compute_from_dx(&u, dx, Direction::Forward, SEQ).unwrap();
        let par = UpwindGradient::compute_from_dx(&u, dx, Direction::Forward, PAR).unwrap();
        assert_eq!(seq, par);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn centered_gradient_parallel_matches_sequential() {
        let m = mesh(64);
        let dx = m.characteristic_length();
        let u = DVector::from_vec((0..64).map(|i| (i as f64 * dx).sin()).collect());
        let seq = CenteredGradient::compute_from_dx(&u, dx, SEQ).unwrap();
        let par = CenteredGradient::compute_from_dx(&u, dx, PAR).unwrap();
        assert_eq!(seq, par);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn centered_laplacian_parallel_matches_sequential() {
        let m = mesh(64);
        let dx = m.characteristic_length();
        let u = DVector::from_vec((0..64).map(|i| (i as f64 * dx).sin()).collect());
        let seq = CenteredLaplacian::compute_from_dx(&u, dx, SEQ).unwrap();
        let par = CenteredLaplacian::compute_from_dx(&u, dx, PAR).unwrap();
        assert_eq!(seq, par);
    }

    // ── DiscreteOperator::apply — end-to-end via a concrete mesh ──────────────

    #[test]
    fn upwind_gradient_apply_via_discrete_operator() {
        let m = mesh(5);
        let dx = m.characteristic_length();
        let u =
            ContextValue::ScalarField(DVector::from_vec((0..5).map(|i| i as f64 * dx).collect()));
        let op = UpwindGradient::new(Direction::Forward);
        let result = op.apply(&u, &m).unwrap();
        assert!(result
            .as_scalar_field()
            .unwrap()
            .iter()
            .all(|&g| (g - 1.0).abs() < 1e-10));
    }
}
