//! # Module `model`
//!
//! Physical models — `PhysicalModel` trait.
//!
//! A physical model *declares* its context variable needs via `RequiresContext`
//! and *computes* field `u` derivatives at each time step. It does not configure
//! the solving nor orchestrate the time loop — those are the responsibilities of
//! `SolverConfiguration` and `Solver`.
