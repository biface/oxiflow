//! # Module `context::calculators::integral`
//!
//! Spatial quadrature calculator using the trapezoidal rule.

use std::sync::Arc;

use crate::context::calculator::ContextCalculator;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;

// ── TrapezoidalIntegral ───────────────────────────────────────────────────────

/// Computes the spatial integral of the primary field using the trapezoidal rule.
///
/// $$I = \int_\Omega u \, dx \approx \sum_{i=0}^{n-2} \frac{u_i + u_{i+1}}{2} \Delta x_i$$
///
/// where $\Delta x_i = x_{i+1} - x_i$ is taken from the mesh coordinates.
/// This works correctly for both uniform and non-uniform 1D meshes.
///
/// The result is exposed under a user-chosen `ContextVariable::External { name }`,
/// typically `"spatial_mean"`, `"total_mass"`, or similar.
///
/// # Limitations (J2)
///
/// Only 1D meshes are supported at this milestone. The calculator returns
/// `PreconditionFailed` for meshes with `spatial_dimension() != 1`.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use std::borrow::Cow;
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::context::calculators::TrapezoidalIntegral;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
/// use nalgebra::DVector;
///
/// let mesh = Arc::new(UniformGrid1D::new(5, 0.0, 1.0).unwrap());
/// let var  = ContextVariable::External { name: Cow::Borrowed("total_mass") };
/// let calc = TrapezoidalIntegral::new(mesh, var);
///
/// // u = 1  →  ∫₀¹ 1 dx = 1.0
/// let u = DVector::from_element(5, 1.0);
/// let ctx = ComputeContext::new(0.0, 0.01);
/// let val = calc.compute(&ContextValue::ScalarField(u), &ctx).unwrap();
/// assert!((val.as_scalar().unwrap() - 1.0).abs() < 1e-10);
/// ```
pub struct TrapezoidalIntegral {
    mesh: Arc<dyn Mesh>,
    variable: ContextVariable,
}

impl TrapezoidalIntegral {
    /// Creates a new trapezoidal integral calculator.
    ///
    /// # Arguments
    ///
    /// - `mesh` — shared mesh reference (INV-1 compliant).
    /// - `variable` — the `ContextVariable` under which the integral is stored,
    ///   typically `ContextVariable::External { name: "total_mass".into() }`.
    pub fn new(mesh: Arc<dyn Mesh>, variable: ContextVariable) -> Self {
        Self { mesh, variable }
    }
}

impl std::fmt::Debug for TrapezoidalIntegral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrapezoidalIntegral")
            .field("variable", &self.variable)
            .field("mesh_n_dof", &self.mesh.n_dof())
            .finish()
    }
}

impl RequiresContext for TrapezoidalIntegral {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    fn priority(&self) -> u32 {
        100
    }
}

impl ContextCalculator for TrapezoidalIntegral {
    fn provides(&self) -> ContextVariable {
        self.variable.clone()
    }

    fn compute(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        if self.mesh.spatial_dimension() != 1 {
            return Err(OxiflowError::PreconditionFailed {
                context: "TrapezoidalIntegral",
                message: format!(
                    "only 1D meshes are supported at J2, got spatial_dimension = {}",
                    self.mesh.spatial_dimension()
                ),
            });
        }

        let u = state.as_scalar_field()?;
        let n = u.len();

        if n < 2 {
            return Err(OxiflowError::PreconditionFailed {
                context: "TrapezoidalIntegral",
                message: format!("field must have at least 2 nodes, got {n}"),
            });
        }

        let mut integral = 0.0_f64;
        for i in 0..n - 1 {
            let x_i = self.mesh.coordinates(i)[0];
            let x_next = self.mesh.coordinates(i + 1)[0];
            let dx = x_next - x_i;
            integral += (u[i] + u[i + 1]) * 0.5 * dx;
        }

        Ok(ContextValue::Scalar(integral))
    }

    fn name(&self) -> &str {
        "trapezoidal_integral (built-in)"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use nalgebra::DVector;

    use super::*;
    use crate::mesh::UniformGrid1D;

    fn grid(n: usize) -> Arc<dyn Mesh> {
        Arc::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    fn var() -> ContextVariable {
        ContextVariable::External {
            name: Cow::Borrowed("mass"),
        }
    }

    fn ctx() -> ComputeContext {
        ComputeContext::new(0.0, 0.01)
    }

    // ── provides / priority ───────────────────────────────────────────────────

    #[test]
    fn provides_configured_variable() {
        let v = var();
        let calc = TrapezoidalIntegral::new(grid(5), v.clone());
        assert_eq!(calc.provides(), v);
    }

    #[test]
    fn priority_is_one_hundred() {
        let calc = TrapezoidalIntegral::new(grid(5), var());
        assert_eq!(calc.priority(), 100);
    }

    // ── constant field ────────────────────────────────────────────────────────

    #[test]
    fn integral_of_constant_field_equals_domain_length() {
        // u = 1 on [0, 1]  →  ∫₀¹ 1 dx = 1.0
        let n = 11;
        let calc = TrapezoidalIntegral::new(grid(n), var());
        let u = DVector::from_element(n, 1.0);
        let val = calc.compute(&ContextValue::ScalarField(u), &ctx()).unwrap();
        assert!((val.as_scalar().unwrap() - 1.0).abs() < 1e-10);
    }

    // ── linear field: exact for trapezoidal ───────────────────────────────────

    #[test]
    fn integral_of_linear_field_is_exact() {
        // u = x on [0, 1]  →  ∫₀¹ x dx = 0.5  (trapezoidal is exact for linears)
        let n = 5;
        let mesh = grid(n);
        let dx = mesh.characteristic_length();
        let u: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();

        let calc = TrapezoidalIntegral::new(mesh, var());
        let val = calc
            .compute(&ContextValue::ScalarField(DVector::from_vec(u)), &ctx())
            .unwrap();
        assert!((val.as_scalar().unwrap() - 0.5).abs() < 1e-10);
    }

    // ── quadratic field: trapezoidal error decays with n ─────────────────────

    #[test]
    fn integral_of_quadratic_converges_to_exact() {
        // u = x²  →  ∫₀¹ x² dx = 1/3
        // Trapezoidal error O(dx²) — use enough nodes for reasonable accuracy.
        let n = 101;
        let mesh = grid(n);
        let dx = mesh.characteristic_length();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * dx).powi(2)).collect();

        let calc = TrapezoidalIntegral::new(mesh, var());
        let val = calc
            .compute(&ContextValue::ScalarField(DVector::from_vec(u)), &ctx())
            .unwrap();
        let exact = 1.0_f64 / 3.0;
        assert!(
            (val.as_scalar().unwrap() - exact).abs() < 1e-4,
            "expected ≈ {exact}, got {}",
            val.as_scalar().unwrap()
        );
    }

    // ── errors ────────────────────────────────────────────────────────────────

    #[test]
    fn error_on_scalar_state() {
        let calc = TrapezoidalIntegral::new(grid(5), var());
        let result = calc.compute(&ContextValue::Scalar(1.0), &ctx());
        assert!(matches!(result, Err(OxiflowError::TypeMismatch { .. })));
    }

    #[test]
    fn error_on_single_node_field() {
        let mesh = Arc::new(UniformGrid1D::new(2, 0.0, 1.0).unwrap());
        let calc = TrapezoidalIntegral::new(mesh, var());
        let result = calc.compute(
            &ContextValue::ScalarField(DVector::from_vec(vec![1.0])),
            &ctx(),
        );
        assert!(matches!(
            result,
            Err(OxiflowError::PreconditionFailed { .. })
        ));
    }
}
