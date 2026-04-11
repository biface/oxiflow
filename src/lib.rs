//! # oxiflow
//!
//! Generic numerical engine for transport, reaction, and diffusion problems.
//!
//! ## The canonical form
//!
//! Every problem solved by oxiflow is expressed as a conservation law or field
//! equation of the form:
//!
//! $$\frac{\partial u}{\partial t} + \nabla \cdot F(u, \nabla u) = S(u, \mathbf{x}, t)$$
//!
//! where:
//!
//! - **`u(x, t)`** — the primary field: concentration, temperature, velocity,
//!   pressure, magnetic flux density, or any other quantity that evolves in
//!   time and space.
//! - **`F(u, ∇u)`** — the flux tensor, which may combine:
//!   - *advective* flux `v·u` (transport by a flow field),
//!   - *diffusive* flux `-D·∇u` (Fick, Fourier, Darcy),
//!   - *dispersive* flux (higher-order spreading, e.g. in chromatography).
//! - **`S(u, x, t)`** — source or reaction term: chemical reactions, heat
//!   generation, gravity, electromagnetic forcing, adsorption isotherms.
//!
//! Second-order-in-time problems (`∂²u/∂t²`, structural dynamics, wave
//! propagation) are reduced to this form by introducing `v = ∂u/∂t` as a
//! second field component: the state `(u, v)` evolves as a first-order
//! system. Newmark-β and HHT-α integrators are planned for J4+.
//!
//! ## Physical domains
//!
//! oxiflow covers any problem expressible as the canonical form above.
//!
//! | Domain | Primary field `u` | Milestone |
//! |---|---|---|
//! | Chromatographic transport | concentration `c(z,t)` | J1 |
//! | Transient heat conduction | temperature `T(x,t)` | J2+ |
//! | Reaction-diffusion | species `(u,v)` | J2+ |
//! | Flow in porous media (Darcy) | water content `θ` | J2+ |
//! | Shallow water (Saint-Venant) | depth `h`, momentum `hv` | J3 |
//! | Electrochemical (Nernst-Planck) | ionic concentrations `cᵢ` | J3+ |
//! | Resistive magnetic diffusion | flux density `B` | J7 |
//! | Structural dynamics | displacement `u`, velocity `v` | J4+ |
//!
//! For full equations, physical derivations, and numerical details,
//! see the **[project wiki](https://github.com/biface/oxiflow/wiki)**.
//!
//! ## Numerical structure
//!
//! In each domain above, `u` changes physical meaning, but the engine abstraction
//! is identical:
//!
//! | Physical concept | Engine abstraction |
//! |---|---|
//! | Field `u(x,t)` | `ContextValue::ScalarField` / `VectorField` |
//! | Spatial domain | `Mesh` (INV-1) — `UniformGrid1D` (J1), FEM (J7) |
//! | Flux F, source S | `PhysicalModel::compute_physics(u, ctx)` |
//! | Auxiliary quantities | `ComputeContext` ← `ContextCalculator` chain |
//! | Time integration | `Solver` — Euler, RK4, … (J4) |
//! | Boundary conditions | `BoundaryCondition: RequiresContext` (J2) |
//! | Multi-domain coupling | `CouplingOperator` (INV-3, J3) |
//! | Spatial operators | `DiscreteOperator<M: Mesh>` (INV-2, J4b) |
//!
//! ## Architecture — WHAT / HOW separation
//!
//! Three strictly separated responsibilities:
//!
//! | Type | Pole | Role |
//! |---|---|---|
//! | [`solver::Scenario`] | WHAT | Declares the problem: model, mesh, domains, BCs |
//! | [`solver::SolverConfiguration`] | HOW | Configures solving: integrator, step control, calculators |
//! | [`solver::Solver`] | Execution | Orchestrates the time integration loop |
//!
//! The engine enforces a **contractual execution order** at each time step:
//!
//! 1. Context calculators populate [`context::ComputeContext`]
//! 2. Boundary conditions are applied to the current state (J2)
//! 3. [`model::PhysicalModel::compute`] computes `du/dt`
//! 4. The integrator advances the state by `dt`
//!
//! ## Modules
//!
//! - [`context`]   — [`context::ContextVariable`], [`context::ContextValue`],
//!   [`context::OxiflowError`], [`context::ComputeContext`], [`context::ContextCalculator`]
//! - [`mesh`]      — [`mesh::Mesh`] trait (INV-1) and [`mesh::UniformGrid1D`] (J1)
//! - [`model`]     — [`model::RequiresContext`] and [`model::PhysicalModel`]
//! - [`solver`]    — [`solver::Scenario`], [`solver::SolverConfiguration`],
//!   [`solver::Solver`], [`solver::SimulationResult`]
//! - [`boundary`]  — `BoundaryCondition` trait (J2)
//! - [`operators`] — `DiscreteOperator<M: Mesh>` (INV-2, J4b)
//! - [`coupling`]  — `CouplingOperator` (INV-3, J3)
//!
//! ## FEM invariants — forward compatibility
//!
//! All abstractions from v0.1 are designed to extend naturally to FEM at
//! v2.0 without any breaking change on existing code.
//!
//! | Invariant | Guarantee | Active from |
//! |---|---|---|
//! | INV-1 | No `dx`/`nx` in public API — all spatial access via [`mesh::Mesh`] | v0.1.0 |
//! | INV-2 | Integrators decoupled from spatial scheme via `DiscreteOperator<M>` | v0.5.0 |
//! | INV-3 | Inter-domain coupling only via `CouplingOperator` | v0.3.0 |
//! | INV-4 | All public traits object-safe — external crates can implement them | v2.0.0 |
//!
//! ## Getting started (J1)
//!
//! ```rust,ignore
//! use oxiflow::{
//!     context::ContextVariable,
//!     mesh::{Mesh, UniformGrid1D},
//!     model::{PhysicalModel, RequiresContext},
//!     solver::{Scenario, SolverConfiguration, TimeConfiguration,
//!              StepControl, IntegratorKind},
//! };
//!
//! // 1. Declare the physical model
//! struct MyModel;
//! impl RequiresContext for MyModel {
//!     fn required_variables(&self) -> Vec<ContextVariable> {
//!         vec![ContextVariable::Time]
//!     }
//! }
//! impl PhysicalModel for MyModel { /* … */ }
//!
//! // 2. Declare the problem (WHAT)
//! let mesh = UniformGrid1D::new(100, 0.0, 1.0).unwrap();
//! let scenario = Scenario::single(Box::new(MyModel), Box::new(mesh));
//!
//! // 3. Configure solving (HOW)
//! let config = SolverConfiguration::new(
//!     TimeConfiguration::new(600.0, StepControl::Fixed { dt: 0.1 }),
//!     IntegratorKind::Euler,
//! );
//!
//! // 4. Solve (J4 — integrator implementations land at v0.4.0)
//! // let result = EulerSolver.solve(&scenario, &config)?;
//! ```

pub mod boundary;
pub mod context;
pub mod coupling;
pub mod mesh;
pub mod model;
pub mod operators;
pub mod solver;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {}
}
