//! # Module `solver::chain`
//!
//! Solver chain — composition and scheduling of solving steps.
//!
//! Allows composing multiple solving steps (context computation, temporal
//! integration, boundary condition update) into a sequence executable by the `Solver`.
