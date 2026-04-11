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
//! ### Chromatographic transport (J1 validation scenario)
//!
//! A solute migrating through a packed column under applied flow velocity `v`:
//!
//! $$\frac{\partial c}{\partial t} + v\frac{\partial c}{\partial z} = D_{ax}\frac{\partial^2 c}{\partial z^2} - F_{\text{ads}}(c)$$
//!
//! - `u = c(z,t)` — concentration profile along the column axis
//! - `F = v·c − D_ax·∂c/∂z` — convective + axial dispersion flux
//! - `S = −F_ads(c)` — adsorption term (Langmuir, SMA, Thomas isotherm)
//! - Boundary conditions: Danckwerts (Robin inlet + Neumann outlet)
//!   or simplified Dirichlet
//!
//! ### Transient heat conduction
//!
//! Heat diffusion in a solid or fluid medium:
//!
//! $$\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (\lambda \nabla T) + Q(\mathbf{x}, t)$$
//!
//! - `u = T(x, t)` — temperature field
//! - `F = −λ·∇T` — Fourier conductive flux
//! - `S = Q` — internal heat source (Joule heating, chemical reaction)
//! - Boundary conditions: Dirichlet (fixed temperature), Neumann (fixed flux),
//!   Robin (convective exchange `h·(T − T_∞)`)
//!
//! ### Reaction-diffusion systems
//!
//! Two or more coupled species with local reactions (Turing patterns,
//! Gray-Scott, FitzHugh-Nagumo, oscillating chemistry):
//!
//! $$\frac{\partial u}{\partial t} = D_u \nabla^2 u + f(u,v), \qquad \frac{\partial v}{\partial t} = D_v \nabla^2 v + g(u,v)$$
//!
//! - `u = (u, v)` — two interacting field components
//! - `F = −D·∇u` — isotropic diffusion, no advection
//! - `S = f(u, v)` — nonlinear local kinetics
//! - Applications: pattern formation in biology, autocatalytic chemistry
//!
//! ### Flow in porous media (Darcy / Richards)
//!
//! Saturated or unsaturated flow driven by a pressure gradient:
//!
//! $$\frac{\partial \theta}{\partial t} = \nabla \cdot \bigl(K(\theta)\,\nabla\psi\bigr)$$
//!
//! - `u = θ` — volumetric water content (or hydraulic head)
//! - `F = −K(θ)·∇ψ` — Darcy flux, nonlinear conductivity
//! - `S = 0` (or rainfall/extraction source term)
//! - Applications: groundwater, soil contamination, CO₂ storage
//!
//! ### Shallow water flow — gravitational transport
//!
//! Depth-averaged flow on a free surface (Saint-Venant equations):
//!
//! $$\frac{\partial h}{\partial t} + \nabla \cdot (h\mathbf{v}) = 0$$
//!
//! $$\frac{\partial (h\mathbf{v})}{\partial t} + \nabla \cdot (h\mathbf{v} \otimes \mathbf{v}) + g h \nabla \eta = \frac{1}{\rho}\,\boldsymbol{\tau}_b$$
//!
//! - `u = (h, h·v)` — water depth and depth-averaged momentum
//! - `F` — advective + hydrostatic pressure flux
//! - `S` — bed friction `τ_b`, rainfall, infiltration
//! - Applications: lahars, tsunamis, dam breaks, coastal flooding
//!
//! ### Electrochemical transport (Nernst-Planck)
//!
//! Migration and diffusion of ionic species in an electrolyte:
//!
//! $$\frac{\partial c_i}{\partial t} = \nabla \cdot \!\left(D_i \nabla c_i + \frac{z_i D_i}{RT}\,c_i \nabla \varphi\right) + R_i(\mathbf{c})$$
//!
//! - `u = (cᵢ, φ)` — ionic concentrations and electric potential
//! - `F` — diffusive + electromigration flux
//! - `S = Rᵢ(c)` — electrochemical reaction at interfaces
//! - Applications: batteries, fuel cells, electrodeposition
//!
//! ### Resistive magnetic diffusion
//!
//! Low-frequency electromagnetic field in a conducting medium (eddy currents):
//!
//! $$\frac{\partial B}{\partial t} = \frac{1}{\mu\sigma}\,\nabla^2 B$$
//!
//! - `u = B(x, t)` — magnetic flux density
//! - `F = −(1/μσ)·∇B` — resistive diffusion flux
//! - `S = 0` — no external source in the resistive limit
//! - Requires FEM on unstructured meshes for realistic geometries (v2.0, J7)
//!
//! ### Structural dynamics (second-order systems)
//!
//! Vibrations, wave propagation, and dynamic elasticity:
//!
//! $$\frac{\partial^2 u}{\partial t^2} + C(\mathbf{x})\frac{\partial u}{\partial t} + K u = F(\mathbf{x}, t)$$
//!
//! Reduced to first-order form by introducing `v = ∂u/∂t`:
//!
//! $$\frac{\partial u}{\partial t} = v, \qquad \frac{\partial v}{\partial t} = F - Cv - Ku$$
//!
//! - `u = (displacement, velocity)` — two-component state vector
//! - `S` — applied forces, damping
//! - Integrators: Newmark-β, HHT-α (planned J4+)
//! - Applications: structural analysis, seismic response, acoustics
//!
//! ### Spectral methods — planned extension (DD-024)
//!
//! Spectral methods (Fourier, Chebyshev, Legendre, spherical harmonics) offer
//! exponential convergence for smooth solutions. They approximate `u` as a sum
//! of global basis functions rather than nodal values:
//!
//! $$u(\mathbf{x}, t) \approx \sum_n \hat{u}_n(t)\,\varphi_n(\mathbf{x})$$
//!
//! Spatial operators become matrix multiplications in spectral space
//! (`∂u/∂x → D·û`) rather than finite differences between neighbours.
//! This yields accuracy comparable to thousands of FD nodes with only tens
//! of spectral modes for smooth profiles.
//!
//! Their integration raises an open architectural question (DD-024): whether
//! a `SpectralBasis` should implement the existing `Mesh` trait or introduce
//! a parallel `Basis` trait alongside it. This decision is deferred to J7+.
//!
//! Relevant domains: global atmospheric modelling (spherical harmonics),
//! turbulence DNS, high-precision chromatography, acoustics.
//!
//! ## Numerical structure
//!
//! In each case above, `u` changes physical meaning, but the numerical
//! structure is identical: a field discretised on a mesh, fluxes computed
//! at interfaces or nodes, time integration. This is what the engine abstracts:
//!
//! ```text
//! Physical concept           Engine abstraction
//! ─────────────────          ──────────────────────────────────────────
//! field u(x, t)         →    ContextValue::ScalarField / VectorField
//! spatial domain        →    Mesh (INV-1) — UniformGrid1D (J1), FEM (J7)
//! flux F, source S      →    PhysicalModel::compute_physics(u, ctx)
//! auxiliary quantities  →    ComputeContext ← ContextCalculator chain
//! time integration      →    Solver — Euler, RK4, Newmark… (J4, J4+)
//! boundary conditions   →    BoundaryCondition: RequiresContext (J2)
//! multi-domain coupling →    CouplingOperator (INV-3, J3)
//! spatial operators     →    DiscreteOperator<M: Mesh> (INV-2, J4b)
//! ```
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
//! 3. [`model::PhysicalModel::compute_physics`] computes `du/dt`
//! 4. The integrator advances the state by `dt`
//!
//! ## Modules
//!
//! - [`context`]   — [`context::ContextVariable`], [`context::ContextValue`],
//!   [`context::OxiflowError`], [`context::ComputeContext`],
//!   [`context::ContextCalculator`]
//! - [`mesh`]      — [`mesh::Mesh`] trait (INV-1) and
//!   [`mesh::UniformGrid1D`] (J1)
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
