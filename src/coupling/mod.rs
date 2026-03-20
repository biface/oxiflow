//! # Module `coupling`
//!
//! Multi-physics inter-domain coupling — INV-3 invariant.
//!
//! ## Core principle (INV-3)
//!
//! All interactions between distinct physical domains (lahar/lake, fluid/solid,
//! column/reservoir) must go through the `CouplingOperator` trait with `DomainId`
//! and `Interface`. No coupling logic is coded directly in the `Solver`
//! (DD-011, J3).
