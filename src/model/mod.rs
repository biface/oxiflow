//! # Module `model`
//!
//! Physical models — `PhysicalModel` trait and component traits.
//!
//! A physical model *declares* its context variable needs via `RequiresContext`
//! and *computes* field `u` derivatives at each time step. It does not configure
//! the solving nor orchestrate the time loop — those are the responsibilities of
//! `SolverConfiguration` and `Solver`.

pub mod composite;
pub mod traits;

pub use composite::CompositeModel;
pub use traits::PhysicalModel;
pub use traits::RequiresContext;
