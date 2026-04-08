//! # Module `mesh::structured`
//!
//! Structured mesh implementations — uniform grids for FD/FV schemes (DD-019).
//!
//! ## J1 (v0.1.0)
//!
//! - [`UniformGrid1D`] — 1D uniform grid, the first `Mesh` implementation.
//!
//! ## Future (J4b+)
//!
//! - `UniformGrid2D` — 2D Cartesian grid for 2D FD/FV problems.

use crate::mesh::Mesh;

/// 1D uniform grid — first concrete implementation of [`Mesh`] (INV-1, J1).
///
/// Nodes are evenly spaced at positions `x_i = x_start + i * dx` for
/// `i = 0..n_points`. The coordinate table is pre-computed at construction
/// so that `coordinates(i)` returns a zero-allocation slice.
///
/// # Invariant compliance (INV-1)
///
/// `dx` and `n_points` are private. Callers access spatial information only
/// through the `Mesh` trait methods — `n_dof()`, `coordinates()`,
/// `spatial_dimension()`, `characteristic_length()`.
///
/// # Examples
///
/// ```rust
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
///
/// let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
///
/// assert_eq!(grid.n_dof(), 5);
/// assert_eq!(grid.spatial_dimension(), 1);
/// assert!((grid.characteristic_length() - 0.25).abs() < 1e-12);
///
/// // Coordinates: 0.0, 0.25, 0.50, 0.75, 1.0
/// assert!((grid.coordinates(0)[0] - 0.0).abs() < 1e-12);
/// assert!((grid.coordinates(2)[0] - 0.5).abs() < 1e-12);
/// assert!((grid.coordinates(4)[0] - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug)]
pub struct UniformGrid1D {
    /// Number of nodes.
    n_points: usize,
    /// Spatial step — private per INV-1.
    dx: f64,
    /// Pre-computed node coordinates, length = n_points.
    nodes: Vec<[f64; 1]>,
}

impl UniformGrid1D {
    /// Creates a uniform 1D grid with `n_points` nodes from `x_start` to `x_end`.
    ///
    /// Node positions: `x_i = x_start + i * dx` where `dx = (x_end - x_start) / (n_points - 1)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `n_points < 2` — a grid must have at least 2 nodes.
    /// - `x_end <= x_start` — the domain must be non-degenerate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxiflow::mesh::{Mesh, UniformGrid1D};
    ///
    /// let grid = UniformGrid1D::new(100, 0.0, 1.0).unwrap();
    /// assert_eq!(grid.n_dof(), 100);
    ///
    /// assert!(UniformGrid1D::new(1, 0.0, 1.0).is_err()); // n_points < 2
    /// assert!(UniformGrid1D::new(10, 1.0, 0.0).is_err()); // x_end <= x_start
    /// ```
    pub fn new(n_points: usize, x_start: f64, x_end: f64) -> Result<Self, String> {
        if n_points < 2 {
            return Err(format!(
                "UniformGrid1D requires at least 2 nodes, got {}",
                n_points
            ));
        }
        if x_end <= x_start {
            return Err(format!(
                "UniformGrid1D requires x_end > x_start, got [{}, {}]",
                x_start, x_end
            ));
        }

        let dx = (x_end - x_start) / (n_points - 1) as f64;
        let nodes = (0..n_points).map(|i| [x_start + i as f64 * dx]).collect();

        Ok(Self {
            n_points,
            dx,
            nodes,
        })
    }
}

impl Mesh for UniformGrid1D {
    fn n_dof(&self) -> usize {
        self.n_points
    }

    /// Returns the coordinate of node `i` as a single-element slice `[x_i]`.
    ///
    /// Zero allocation — the slice points into the pre-computed `nodes` table.
    ///
    /// # Panics
    ///
    /// Panics if `i >= n_dof()`.
    fn coordinates(&self, i: usize) -> &[f64] {
        &self.nodes[i]
    }

    fn spatial_dimension(&self) -> usize {
        1
    }

    /// Returns `dx` — the uniform node spacing.
    ///
    /// Used by integrators for CFL and Péclet estimates.
    fn characteristic_length(&self) -> f64 {
        self.dx
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn new_valid_grid_succeeds() {
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        assert_eq!(grid.n_dof(), 5);
    }

    #[test]
    fn new_with_one_node_fails() {
        assert!(UniformGrid1D::new(1, 0.0, 1.0).is_err());
    }

    #[test]
    fn new_with_zero_nodes_fails() {
        assert!(UniformGrid1D::new(0, 0.0, 1.0).is_err());
    }

    #[test]
    fn new_with_reversed_domain_fails() {
        assert!(UniformGrid1D::new(10, 1.0, 0.0).is_err());
    }

    #[test]
    fn new_with_equal_bounds_fails() {
        assert!(UniformGrid1D::new(10, 0.5, 0.5).is_err());
    }

    #[test]
    fn error_message_mentions_constraint() {
        let err = UniformGrid1D::new(1, 0.0, 1.0).unwrap_err();
        assert!(err.contains("2"));
        let err = UniformGrid1D::new(10, 1.0, 0.0).unwrap_err();
        assert!(err.contains("x_end"));
    }

    // ── n_dof ─────────────────────────────────────────────────────────────────

    #[test]
    fn n_dof_equals_n_points() {
        for n in [2, 10, 100, 1000] {
            let grid = UniformGrid1D::new(n, 0.0, 1.0).unwrap();
            assert_eq!(grid.n_dof(), n);
        }
    }

    // ── coordinates ───────────────────────────────────────────────────────────

    #[test]
    fn first_node_at_x_start() {
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        assert!((grid.coordinates(0)[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn last_node_at_x_end() {
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        assert!((grid.coordinates(4)[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn middle_node_correctly_positioned() {
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        // dx = 0.25 → node 2 at 0.5
        assert!((grid.coordinates(2)[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn coordinates_with_non_zero_x_start() {
        let grid = UniformGrid1D::new(3, 1.0, 3.0).unwrap();
        // dx = 1.0 → nodes at 1.0, 2.0, 3.0
        assert!((grid.coordinates(0)[0] - 1.0).abs() < 1e-12);
        assert!((grid.coordinates(1)[0] - 2.0).abs() < 1e-12);
        assert!((grid.coordinates(2)[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn coordinates_slice_has_length_one() {
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        for i in 0..grid.n_dof() {
            assert_eq!(grid.coordinates(i).len(), 1);
        }
    }

    #[test]
    fn coordinates_are_strictly_increasing() {
        let grid = UniformGrid1D::new(10, 0.0, 1.0).unwrap();
        for i in 1..grid.n_dof() {
            assert!(grid.coordinates(i)[0] > grid.coordinates(i - 1)[0]);
        }
    }

    #[test]
    fn coordinates_zero_allocation_returns_reference() {
        // Verify coordinates() returns a slice into the node table,
        // not a newly allocated Vec — checked by testing two calls return
        // the same pointer for the same index.
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        let ptr1 = grid.coordinates(2).as_ptr();
        let ptr2 = grid.coordinates(2).as_ptr();
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    #[should_panic]
    fn coordinates_out_of_bounds_panics() {
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        let _ = grid.coordinates(5); // index == n_dof → panic
    }

    // ── spatial_dimension ─────────────────────────────────────────────────────

    #[test]
    fn spatial_dimension_is_one() {
        let grid = UniformGrid1D::new(10, 0.0, 1.0).unwrap();
        assert_eq!(grid.spatial_dimension(), 1);
    }

    // ── characteristic_length ─────────────────────────────────────────────────

    #[test]
    fn characteristic_length_equals_dx() {
        // 5 nodes over [0, 1] → dx = 0.25
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        assert!((grid.characteristic_length() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn characteristic_length_halves_when_doubling_nodes() {
        let coarse = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        let fine = UniformGrid1D::new(9, 0.0, 1.0).unwrap();
        let ratio = coarse.characteristic_length() / fine.characteristic_length();
        assert!((ratio - 2.0).abs() < 1e-12);
    }

    #[test]
    fn characteristic_length_scales_with_domain() {
        let unit = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        let large = UniformGrid1D::new(5, 0.0, 4.0).unwrap();
        let ratio = large.characteristic_length() / unit.characteristic_length();
        assert!((ratio - 4.0).abs() < 1e-12);
    }

    // ── Mesh trait via dyn ────────────────────────────────────────────────────

    #[test]
    fn usable_as_dyn_mesh() {
        let grid: Box<dyn Mesh> = Box::new(UniformGrid1D::new(10, 0.0, 1.0).unwrap());
        assert_eq!(grid.n_dof(), 10);
        assert_eq!(grid.spatial_dimension(), 1);
        assert!((grid.characteristic_length() - 1.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn mesh_vec_of_different_grids() {
        let grids: Vec<Box<dyn Mesh>> = vec![
            Box::new(UniformGrid1D::new(10, 0.0, 1.0).unwrap()),
            Box::new(UniformGrid1D::new(50, 0.0, 0.25).unwrap()),
        ];
        assert_eq!(grids[0].n_dof(), 10);
        assert_eq!(grids[1].n_dof(), 50);
    }

    // ── Send + Sync ───────────────────────────────────────────────────────────

    #[test]
    fn uniform_grid_1d_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<UniformGrid1D>();
    }

    // ── INV-1 — no dx/n_points in public API ──────────────────────────────────

    #[test]
    fn inv1_dx_not_accessible_directly() {
        // This test documents INV-1 compliance: the only way to get dx is
        // via characteristic_length(). There is no public `grid.dx` field.
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        let dx = grid.characteristic_length(); // the only correct path
        assert!((dx - 0.25).abs() < 1e-12);
    }

    #[test]
    fn inv1_n_points_not_accessible_directly() {
        // The only way to get the number of nodes is via n_dof().
        let grid = UniformGrid1D::new(7, 0.0, 1.0).unwrap();
        assert_eq!(grid.n_dof(), 7);
    }
}
