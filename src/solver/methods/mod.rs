//! # Module `solver::methods`
//!
//! *Temporal integration methods — J4a (v0.5).*
//!
//! ## Planned methods
//!
//! | Méthode | Type | Milestone |
//! |---|---|---|
//! | Euler explicite | Explicit | J4a |
//! | RK4 | Explicit | J4a |
//! | DoPri45 | Adaptive | J4a |
//! | Euler implicite | Implicit | J4a |
//! | Crank–Nicolson | Semi-implicit | J4a |
//! | BDF2/3 | Implicit multi-step | J4a |
//! | IMEX (splitting de Strang) | Hybrid | J4a |
//!
//! Integrators are generic over `DiscreteOperator<M: Mesh>` (INV-2) — no spatial
//! scheme is called directly inside an integrator.
