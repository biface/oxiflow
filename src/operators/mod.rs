//! # Module `operators`
//!
//! Spatial discretization operators — INV-2 invariant.
//!
//! ## Core principle (INV-2)
//!
//! Temporal integrators are decoupled from the spatial discretization scheme via
//! the `DiscreteOperator<M: Mesh>` trait. No method — FD, FV or FEM — is called
//! directly in an integrator. Each scheme is a separate trait implementation
//! (DD-012, J4b).
//!
//! ## Planned implementations
//!
//! | Type | Scheme | Milestone |
//! |---|---|---|
//! | `FiniteDifference1D` | Upwind/centered FD | J4b — v0.6 |
//! | `FiniteVolume` | Conservative FV + MinMod | J4b — v0.6 |
//! | `WENO5` | WENO 5 order | J4b — v0.6 |
//! | `FiniteElement` | FEM P1/P2 on mesh | J7 — v2.0 |
