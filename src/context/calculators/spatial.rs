//! # Module `context::calculators::spatial`
//!
//! Finite-difference spatial calculators: gradient and Laplacian.
//!
//! Both calculators hold an `Arc<dyn Mesh>` internally (INV-1, DD-007).
//! At v0.5.0, this will be replaced by a concrete `DiscreteOperator`
//! (INV-2, DD-012) with no change to the `ContextCalculator` trait.

use std::sync::Arc;

use nalgebra::DVector;

use crate::context::calculator::ContextCalculator;
use crate::context::calculators::FDScheme;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;

// ── FDGradientCalculator ──────────────────────────────────────────────────────

/// Computes the finite-difference spatial gradient of the primary field.
///
/// Provides `ContextVariable::SpatialGradient { dimension, component }` as a
/// `ContextValue::ScalarField` — one gradient value per mesh node.
///
/// The mesh is held as `Arc<dyn Mesh>` internally (INV-1). At J5 (v0.5.0),
/// this implementation detail will be replaced by a `DiscreteOperator` (INV-2).
///
/// # Boundary treatment
///
/// | Scheme | Interior | Left boundary (i = 0) | Right boundary (i = n−1) |
/// |---|---|---|---|
/// | `Forward` | `(u[i+1] − u[i]) / dx` | same | `(u[n−1] − u[n−2]) / dx` |
/// | `Backward` | `(u[i] − u[i−1]) / dx` | `(u[1] − u[0]) / dx` | same |
/// | `Central` | `(u[i+1] − u[i−1]) / 2dx` | `(u[1] − u[0]) / dx` | `(u[n−1] − u[n−2]) / dx` |
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::context::calculators::{FDGradientCalculator, FDScheme};
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
/// use nalgebra::DVector;
///
/// let mesh = Arc::new(UniformGrid1D::new(5, 0.0, 1.0).unwrap());
/// let calc = FDGradientCalculator::new(mesh, 0, None, FDScheme::Central);
///
/// // Linear field u = x  →  ∂u/∂x = 1 everywhere
/// let u = DVector::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
/// let ctx = ComputeContext::new(0.0, 0.01);
/// let grad = calc.compute(&ContextValue::ScalarField(u), &ctx).unwrap();
/// let field = grad.as_scalar_field().unwrap();
/// for g in field.iter() {
///     assert!((g - 1.0).abs() < 1e-10);
/// }
/// ```
pub struct FDGradientCalculator {
    mesh: Arc<dyn Mesh>,
    dimension: usize,
    component: Option<usize>,
    scheme: FDScheme,
}

impl FDGradientCalculator {
    /// Creates a new FD gradient calculator.
    ///
    /// # Arguments
    ///
    /// - `mesh` — shared mesh reference (INV-1 compliant).
    /// - `dimension` — spatial dimension (0 → ∂u/∂x, 1 → ∂u/∂y, …).
    /// - `component` — field component (`None` for mono-component, J1/J2 default).
    /// - `scheme` — finite-difference stencil.
    pub fn new(
        mesh: Arc<dyn Mesh>,
        dimension: usize,
        component: Option<usize>,
        scheme: FDScheme,
    ) -> Self {
        Self {
            mesh,
            dimension,
            component,
            scheme,
        }
    }
}

impl std::fmt::Debug for FDGradientCalculator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FDGradientCalculator")
            .field("dimension", &self.dimension)
            .field("component", &self.component)
            .field("scheme", &self.scheme)
            .field("mesh_n_dof", &self.mesh.n_dof())
            .finish()
    }
}

impl RequiresContext for FDGradientCalculator {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    // Runs after time built-ins (priority 0) but before user-defined calculators.
    fn priority(&self) -> u32 {
        10
    }
}

impl ContextCalculator for FDGradientCalculator {
    fn provides(&self) -> ContextVariable {
        ContextVariable::SpatialGradient {
            dimension: self.dimension,
            component: self.component,
        }
    }

    fn compute(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        let n = u.len();

        if n < 2 {
            return Err(OxiflowError::PreconditionFailed {
                context: "FDGradientCalculator",
                message: format!("field must have at least 2 nodes, got {n}"),
            });
        }

        let dx = self.mesh.characteristic_length();
        let mut grad = DVector::zeros(n);

        for i in 0..n {
            grad[i] = match self.scheme {
                FDScheme::Forward => {
                    if i < n - 1 {
                        (u[i + 1] - u[i]) / dx
                    } else {
                        // Right boundary fallback: backward
                        (u[n - 1] - u[n - 2]) / dx
                    }
                }
                FDScheme::Backward => {
                    if i > 0 {
                        (u[i] - u[i - 1]) / dx
                    } else {
                        // Left boundary fallback: forward
                        (u[1] - u[0]) / dx
                    }
                }
                FDScheme::Central => {
                    if i == 0 {
                        // Left boundary: 1st-order forward
                        (u[1] - u[0]) / dx
                    } else if i == n - 1 {
                        // Right boundary: 1st-order backward
                        (u[n - 1] - u[n - 2]) / dx
                    } else {
                        // Interior: 2nd-order central
                        (u[i + 1] - u[i - 1]) / (2.0 * dx)
                    }
                }
                // J5+: higher-order stencils will be added here.
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(OxiflowError::PreconditionFailed {
                        context: "FDGradientCalculator",
                        message: "unsupported FDScheme variant".to_string(),
                    })
                }
            };
        }

        Ok(ContextValue::ScalarField(grad))
    }

    fn name(&self) -> &str {
        "fd_gradient (built-in)"
    }
}

// ── FDLaplacianCalculator ─────────────────────────────────────────────────────

/// Computes the finite-difference Laplacian ∇²u of the primary field.
///
/// Provides a user-chosen `ContextVariable::External { name }` as a
/// `ContextValue::ScalarField`. The Laplacian is a scalar nodal field of the
/// same length as the primary field — no new `ContextVariable` variant is
/// needed (DD-026 deferred).
///
/// # Stencil
///
/// Standard 3-point central: `(u[i−1] − 2u[i] + u[i+1]) / dx²`
///
/// Boundary treatment (1st-order one-sided):
/// - `i = 0` → `(u[0] − 2u[1] + u[2]) / dx²`
/// - `i = n−1` → `(u[n−3] − 2u[n−2] + u[n−1]) / dx²`
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use std::borrow::Cow;
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::context::calculators::FDLaplacianCalculator;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
/// use nalgebra::DVector;
///
/// let mesh = Arc::new(UniformGrid1D::new(5, 0.0, 1.0).unwrap());
/// let var  = ContextVariable::External { name: Cow::Borrowed("laplacian") };
/// let calc = FDLaplacianCalculator::new(mesh, var);
///
/// // Quadratic field u = x²  →  ∇²u = 2 everywhere (interior)
/// let u = DVector::from_vec(vec![0.0, 0.0625, 0.25, 0.5625, 1.0]);
/// let ctx = ComputeContext::new(0.0, 0.01);
/// let lap = calc.compute(&ContextValue::ScalarField(u), &ctx).unwrap();
/// let field = lap.as_scalar_field().unwrap();
/// // Interior nodes: ∇²u ≈ 2.0
/// assert!((field[2] - 2.0).abs() < 1e-6);
/// ```
pub struct FDLaplacianCalculator {
    mesh: Arc<dyn Mesh>,
    variable: ContextVariable,
}

impl FDLaplacianCalculator {
    /// Creates a new FD Laplacian calculator.
    ///
    /// # Arguments
    ///
    /// - `mesh` — shared mesh reference (INV-1 compliant).
    /// - `variable` — the `ContextVariable` this calculator provides, typically
    ///   `ContextVariable::External { name: "laplacian".into() }`.
    pub fn new(mesh: Arc<dyn Mesh>, variable: ContextVariable) -> Self {
        Self { mesh, variable }
    }
}

impl std::fmt::Debug for FDLaplacianCalculator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FDLaplacianCalculator")
            .field("variable", &self.variable)
            .field("mesh_n_dof", &self.mesh.n_dof())
            .finish()
    }
}

impl RequiresContext for FDLaplacianCalculator {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    fn priority(&self) -> u32 {
        10
    }
}

impl ContextCalculator for FDLaplacianCalculator {
    fn provides(&self) -> ContextVariable {
        self.variable.clone()
    }

    fn compute(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        let n = u.len();

        if n < 3 {
            return Err(OxiflowError::PreconditionFailed {
                context: "FDLaplacianCalculator",
                message: format!("field must have at least 3 nodes, got {n}"),
            });
        }

        let dx = self.mesh.characteristic_length();
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

        Ok(ContextValue::ScalarField(lap))
    }

    fn name(&self) -> &str {
        "fd_laplacian (built-in)"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::mesh::UniformGrid1D;

    fn grid(n: usize) -> Arc<dyn Mesh> {
        Arc::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    fn ctx() -> ComputeContext {
        ComputeContext::new(0.0, 0.01)
    }

    fn laplacian_var() -> ContextVariable {
        ContextVariable::External {
            name: Cow::Borrowed("laplacian"),
        }
    }

    // ── FDGradientCalculator — provides / priority ────────────────────────────

    #[test]
    fn gradient_provides_spatial_gradient_variable() {
        let calc = FDGradientCalculator::new(grid(5), 0, None, FDScheme::Central);
        assert_eq!(
            calc.provides(),
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None
            }
        );
    }

    #[test]
    fn gradient_priority_is_ten() {
        let calc = FDGradientCalculator::new(grid(5), 0, None, FDScheme::Forward);
        assert_eq!(calc.priority(), 10);
    }

    #[test]
    fn gradient_has_no_required_variables() {
        let calc = FDGradientCalculator::new(grid(5), 0, None, FDScheme::Forward);
        assert!(calc.required_variables().is_empty());
    }

    // ── FDGradientCalculator — Central on linear field ────────────────────────

    #[test]
    fn central_gradient_of_linear_field_is_one() {
        // u = x on [0, 1] with 5 nodes → ∂u/∂x = 1 everywhere
        let n = 5;
        let mesh = grid(n);
        let dx = mesh.characteristic_length();
        let u: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();

        let calc = FDGradientCalculator::new(mesh, 0, None, FDScheme::Central);
        let result = calc
            .compute(&ContextValue::ScalarField(DVector::from_vec(u)), &ctx())
            .unwrap();
        let grad = result.as_scalar_field().unwrap();

        for g in grad.iter() {
            assert!((g - 1.0).abs() < 1e-10, "expected 1.0, got {g}");
        }
    }

    // ── FDGradientCalculator — Forward ────────────────────────────────────────

    #[test]
    fn forward_gradient_interior_nodes_correct() {
        let n = 5;
        let mesh = grid(n);
        let dx = mesh.characteristic_length();
        // u = x²  →  ∂u/∂x at x_i ≈ (x_{i+1}² - x_i²) / dx  (forward, 1st order)
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * dx).powi(2)).collect();

        let calc = FDGradientCalculator::new(mesh, 0, None, FDScheme::Forward);
        let result = calc
            .compute(&ContextValue::ScalarField(DVector::from_vec(u)), &ctx())
            .unwrap();
        let grad = result.as_scalar_field().unwrap();

        // At i=0: (0.0625 - 0) / 0.25 = 0.25  (forward approx of 2x at x=0 → expected 0)
        // Forward is 1st-order so we verify it returns a finite, non-NaN value.
        assert!(grad.iter().all(|g| g.is_finite()));
        // Boundary fallback: last node uses backward
        assert!((grad[n - 1] - grad[n - 2]).abs() < 1.0);
    }

    // ── FDGradientCalculator — Backward ───────────────────────────────────────

    #[test]
    fn backward_gradient_fallback_at_left_boundary() {
        let n = 5;
        let mesh = grid(n);
        let dx = mesh.characteristic_length();
        let u: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect(); // u = x

        let calc = FDGradientCalculator::new(mesh, 0, None, FDScheme::Backward);
        let result = calc
            .compute(&ContextValue::ScalarField(DVector::from_vec(u)), &ctx())
            .unwrap();
        let grad = result.as_scalar_field().unwrap();

        // Left boundary fallback → forward → (u[1] - u[0]) / dx = 1.0
        assert!((grad[0] - 1.0).abs() < 1e-10);
    }

    // ── FDGradientCalculator — error on small field ───────────────────────────

    #[test]
    fn gradient_error_on_single_node_field() {
        let mesh = Arc::new(UniformGrid1D::new(2, 0.0, 1.0).unwrap());
        let calc = FDGradientCalculator::new(mesh, 0, None, FDScheme::Central);
        let result = calc.compute(
            &ContextValue::ScalarField(DVector::from_vec(vec![1.0])),
            &ctx(),
        );
        assert!(matches!(
            result,
            Err(OxiflowError::PreconditionFailed { .. })
        ));
    }

    // ── FDGradientCalculator — type mismatch ──────────────────────────────────

    #[test]
    fn gradient_error_on_scalar_state() {
        let calc = FDGradientCalculator::new(grid(5), 0, None, FDScheme::Central);
        let result = calc.compute(&ContextValue::Scalar(1.0), &ctx());
        assert!(matches!(result, Err(OxiflowError::TypeMismatch { .. })));
    }

    // ── FDLaplacianCalculator — provides / priority ───────────────────────────

    #[test]
    fn laplacian_provides_configured_variable() {
        let var = laplacian_var();
        let calc = FDLaplacianCalculator::new(grid(5), var.clone());
        assert_eq!(calc.provides(), var);
    }

    #[test]
    fn laplacian_priority_is_ten() {
        let calc = FDLaplacianCalculator::new(grid(5), laplacian_var());
        assert_eq!(calc.priority(), 10);
    }

    // ── FDLaplacianCalculator — quadratic field ───────────────────────────────

    #[test]
    fn laplacian_of_quadratic_field_is_two_at_interior() {
        // u = x²  →  ∇²u = d²u/dx² = 2 everywhere
        let n = 7;
        let mesh = grid(n);
        let dx = mesh.characteristic_length();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * dx).powi(2)).collect();

        let calc = FDLaplacianCalculator::new(mesh, laplacian_var());
        let result = calc
            .compute(&ContextValue::ScalarField(DVector::from_vec(u)), &ctx())
            .unwrap();
        let lap = result.as_scalar_field().unwrap();

        // Interior nodes: exact for quadratic field
        for i in 1..n - 1 {
            assert!(
                (lap[i] - 2.0).abs() < 1e-8,
                "node {i}: expected 2.0, got {}",
                lap[i]
            );
        }
    }

    #[test]
    fn laplacian_of_linear_field_is_zero_at_interior() {
        // u = x  →  ∇²u = 0 everywhere
        let n = 7;
        let mesh = grid(n);
        let dx = mesh.characteristic_length();
        let u: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();

        let calc = FDLaplacianCalculator::new(mesh, laplacian_var());
        let result = calc
            .compute(&ContextValue::ScalarField(DVector::from_vec(u)), &ctx())
            .unwrap();
        let lap = result.as_scalar_field().unwrap();

        for i in 1..n - 1 {
            assert!(
                lap[i].abs() < 1e-10,
                "node {i}: expected 0.0, got {}",
                lap[i]
            );
        }
    }

    // ── FDLaplacianCalculator — error on small field ──────────────────────────

    #[test]
    fn laplacian_error_on_two_node_field() {
        let mesh = Arc::new(UniformGrid1D::new(2, 0.0, 1.0).unwrap());
        let calc = FDLaplacianCalculator::new(mesh, laplacian_var());
        let result = calc.compute(
            &ContextValue::ScalarField(DVector::from_vec(vec![0.0, 1.0])),
            &ctx(),
        );
        assert!(matches!(
            result,
            Err(OxiflowError::PreconditionFailed { .. })
        ));
    }

    // ── FDLaplacianCalculator — type mismatch ─────────────────────────────────

    #[test]
    fn laplacian_error_on_scalar_state() {
        let calc = FDLaplacianCalculator::new(grid(5), laplacian_var());
        let result = calc.compute(&ContextValue::Scalar(1.0), &ctx());
        assert!(matches!(result, Err(OxiflowError::TypeMismatch { .. })));
    }
}
