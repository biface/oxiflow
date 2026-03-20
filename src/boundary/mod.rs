//! # Module `boundary`
//!
//! Boundary conditions — `BoundaryCondition` trait.
//!
//! From J2 (v0.3), boundary conditions will be able to require context variables
//! by becoming super-traits of `RequiresContext` (DD-008). `Dirichlet`, `Neumann`
//! and `Robin` types (Danckwerts conditions) will be implemented at that stage.
