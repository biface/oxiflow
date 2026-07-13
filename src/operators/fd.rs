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

use nalgebra::DVector;

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::mesh::structured::UniformGrid1D;
use crate::mesh::Mesh;
use crate::operators::DiscreteOperator;

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
    /// documentation for why this bypasses `Self::MeshType`.
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
    /// documentation for why this bypasses `Self::MeshType`.
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
                // Left boundary: 1st-order forward.
                (u[1] - u[0]) / dx
            } else if i == n - 1 {
                // Right boundary: 1st-order backward.
                (u[n - 1] - u[n - 2]) / dx
            } else {
                // Interior: 2nd-order central.
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
    /// documentation for why this bypasses `Self::MeshType`.
    pub(crate) fn compute_from_dx(u: &DVector<f64>, dx: f64) -> Result<DVector<f64>, OxiflowError> {
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

        // Interior: standard 3-point central difference.
        for i in 1..n - 1 {
            lap[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / dx2;
        }

        // Right boundary: one-sided stencil using nodes [n-3, n-2, n-1].
        lap[n - 1] = (u[n - 3] - 2.0 * u[n - 2] + u[n - 1]) / dx2;

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
        let lap = Self::compute_from_dx(u, dx)?;
        Ok(ContextValue::ScalarField(lap))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        let grad = UpwindGradient::compute_from_dx(&u, dx, Direction::Forward).unwrap();
        assert!(grad.iter().all(|&g| (g - 1.0).abs() < 1e-10));
    }

    #[test]
    fn upwind_backward_fallback_at_left_boundary() {
        let m = mesh(5);
        let dx = m.characteristic_length();
        let u = DVector::from_vec((0..5).map(|i| i as f64 * dx).collect());
        let grad = UpwindGradient::compute_from_dx(&u, dx, Direction::Backward).unwrap();
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
                let grad = UpwindGradient::compute_from_dx(&u, dx, Direction::Forward).unwrap();
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
        let grad = CenteredGradient::compute_from_dx(&u, dx).unwrap();
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
                let grad = CenteredGradient::compute_from_dx(&u, dx).unwrap();
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
        let lap = CenteredLaplacian::compute_from_dx(&u, dx).unwrap();
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
                let lap = CenteredLaplacian::compute_from_dx(&u, dx).unwrap();
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
        let err =
            UpwindGradient::compute_from_dx(&DVector::from_vec(vec![1.0]), 0.1, Direction::Forward)
                .unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    #[test]
    fn centered_gradient_rejects_single_node_field() {
        let err =
            CenteredGradient::compute_from_dx(&DVector::from_vec(vec![1.0]), 0.1).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    #[test]
    fn centered_laplacian_rejects_two_node_field() {
        let err = CenteredLaplacian::compute_from_dx(&DVector::from_vec(vec![0.0, 1.0]), 0.1)
            .unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
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
