//! # Module `solver::methods`
//!
//! Temporal integration methods.
//!
//! ## Active at J1
//!
//! | Method | Type | Issue |
//! |---|---|---|
//! | [`euler::ForwardEulerSolver`] | Explicit, 1st order | #33 |
//!
//! ## Reserved — J4 (v0.4.0)
//!
//! | Method | Type | Note |
//! |---|---|---|
//! | `RK4Solver` | Explicit, 4th order | — |
//! | `DoPri45Solver` | Adaptive explicit | `StepControl::Adaptive` |
//! | `BackwardEulerSolver` | Implicit, 1st order | Linear solve via DD-013 |
//! | `CrankNicolsonSolver` | Semi-implicit, 2nd order | — |
//! | `BDF2Solver` | Implicit multi-step, 2nd order | — |
//! | `IMEXSolver` | Strang splitting | Transport-reaction |
//!
//! All integrators are decoupled from the spatial scheme via
//! `DiscreteOperator<M: Mesh>` (INV-2, J4b) — no FD/FV/FEM
//! method is called directly inside an integrator.

pub mod euler;

pub use euler::ForwardEulerSolver;
