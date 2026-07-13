//! # Module `operators`
//!
//! Spatial discretization operators — INV-2 invariant.
//!
//! ## Core principle (INV-2)
//!
//! Spatial discretization is decoupled from the rest of the engine through two
//! sibling traits — never a single generic-over-everything trait, and never a
//! scheme called directly from a consumer:
//!
//! - [`DiscreteOperator`] — raw differential quantities (`∂u/∂x`, `∂²u/∂x²`) that
//!   never need physical parameters or context access; `dx` comes from the mesh
//!   alone. Consumed transparently through [`crate::context::ContextCalculator`]
//!   (e.g. `FDGradientCalculator`) — a model declares a required context
//!   variable and never knows which scheme produced it (DD-012).
//! - [`FluxDivergenceOperator`] — the complete `∇·F(u, ∇u)` term, which does need
//!   physical parameters (advection velocity, diffusion coefficient) that may be
//!   space/time-varying. Carries `ctx: &ComputeContext` and `RequiresContext`
//!   directly on `apply()`. Consumed by `DiscretizedModel<Op>` (DD-038, DD-039).
//!
//! `FluxDivergenceOperator` is a **sibling** trait, not a sub-trait of
//! `DiscreteOperator`: their `apply()` signatures diverge (with/without `ctx`),
//! so a sub-trait relationship could not cleanly reuse the parent method.
//! Forcing `ctx`/`RequiresContext` onto `DiscreteOperator` for every scheme was
//! considered and rejected (DD-012, amendment 2) — it would have imposed that
//! capability on FD, which never needs it.
//!
//! **Selection criterion for future schemes** (FEM, spectral methods, J20):
//! does the scheme need per-call access to physical context, or does it reduce
//! to a local differentiation from mesh data alone? The former implements
//! `FluxDivergenceOperator`; the latter implements `DiscreteOperator`.
//!
//! ## Planned implementations
//!
//! | Type | Scheme | Milestone |
//! |---|---|---|
//! | `FiniteDifference1D` | Upwind/centered FD | J4b — v0.6 |
//! | `FiniteVolume` | Conservative FV + MinMod | J4b — v0.6 |
//! | `WENO5` | WENO 5 order | J4b — v0.6 |
//! | `FiniteElement` | FEM P1/P2 on mesh | J7 — v2.0 |
