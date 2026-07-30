# Changelog

All notable changes to oxiflow are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
oxiflow adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Iterated Newton solver core (`solver::methods::newton`): `NewtonConvergence`
  (`ResidualOnly`/`CombinedOr`/`CombinedAnd`), `JacobianStrategy`
  (`ModifiedFrozen`/`FullNewton`/`ModifiedWithStagnationCheck`), and the
  generic convergence loop shared by theta-method and BDF2 — not yet wired
  into `BackwardEulerSolver`/`CrankNicolsonSolver`/`BDF2Solver` (DD-044, #109,
  #110)
- `OxiflowError::NewtonNotConverged { iterations, residual }` — routed
  through `Solver::on_divergence()` once wired (#110)

## [0.6.0] — 2026-07-21

### Highlights

Sparse linear algebra and persistence: a `faer`-backed sparse solver path for
implicit integrators (`BackwardEulerSolver`, `CrankNicolsonSolver`), a
crash/checkpoint snapshot mechanism, HDF5-backed loading of experimental
tabulated data, XML VTK export for ParaView/VisIt, and the first
config-driven integrator selection DTO (`IntegratorSpec`).

### Added

- `faer`-backed sparse linear solver path (`SparseLinearSolver`,
  `FaerSparseSolver`) for `BackwardEulerSolver`/`CrankNicolsonSolver`,
  feature-gated `sparse` (DD-013, DD-043, #50)
- `SimulationSnapshot`, `SnapshotFormat`, `write_snapshot()`/`read_snapshot()`
  (JSON/bincode) — `src/solver/snapshot.rs` (DD-025 Option B / DD-029, #71)
- `Solver::on_divergence()` — default no-op hook, called automatically before
  `OxiflowError::SolverDivergence` is returned (#71)
- `OxiflowError::Persistence` — snapshot/HDF5/VTK I/O and (de)serialisation
  failures
- `ExternalTabulated::from_hdf5()` — loads tabulated data from an HDF5 file,
  feature-gated `hdf5` (DD-027, #105)
- `SimulationResult::write_vtk()` — XML `.vtu` export for ParaView/VisIt,
  feature-gated `vtk` (DD-027, #78)
- `IntegratorSpec` — external config DTO for `BackwardEulerSolver` +
  `TryFrom<IntegratorSpec> for Box<dyn Solver>`, feature-gated `sparse`
  (DD-027 amendment 1, #104)

### Changed

- `SimulationResult` now derives `Clone`
- `hdf5` feature migrated from the unmaintained `hdf5` crate (0.8, aldanor)
  to the actively-maintained `hdf5-metno` fork (chore, #79)

### Design Decisions

- DD-027 — Results export: VTK as interop pivot, HDF5 for bulk datasets (#75)
- DD-027 amendment 1 — config-driven WHAT/HOW construction from JSON/YAML
- DD-029 — Generalised `SimulationSnapshot`: normal checkpoint/resume (#77)
- DD-043 — `SparseLinearSolver`: sibling trait to `DiscreteOperator` (faer)

## [0.5.0] — 2026-07-17

### Added

- `DiscreteOperator` trait — `operators::fd::{UpwindGradient, CenteredGradient,
  CenteredLaplacian}` (#46, #47, DD-012)
- `FluxDivergenceOperator` trait, sibling to `DiscreteOperator` — computes the full
  `∇·F(u, ∇u)` term, needs physical parameters `DiscreteOperator` deliberately does not
  provide (#97, DD-039)
- `DiscretizedModel<Op: FluxDivergenceOperator>` composite — plugs any
  `FluxDivergenceOperator` into `PhysicalModel` (#93, DD-038)
- `operators::fv::{FVCenteredFlux, FVUpwindFlux}` — finite-volume advection/diffusion,
  cell-average semantics (#48, DD-040)
- `operators::weno::{WENO3, WENO5}` — Jiang-Shu substencils, WENO-Z nonlinear weighting
  (Borges et al. 2008) for critical-point accuracy (#49, DD-041)
- `FluxBoundary` enum — explicit boundary-treatment choice per operator:
  - `Periodic` — exact conservation via telescoping (#48, #49)
  - `Truncation` — one-sided decentered fallback at boundary cells, breaks exact
    conservation (documented trade-off) (DD-042)
  - `GhostCell` — real `BoundaryCondition` physics via `ghost_value(depth,
    interior_at_depth, dx)`, method-of-images convention, exact at every depth for a
    Robin condition (DD-042, amendment 1)
- `BoundaryCondition::ghost_value(depth, interior_at_depth, dx) -> Option<f64>` — default
  method, `None` by default, non-breaking (DD-042)
- `operators::limiters::{Limiter, LimitedFlux}` — `MinMod`/`VanLeer`/`Superbee`, MUSCL
  reconstruction (#99)
- `operators::adaptive::AdaptiveFlux` — blends `WENO3`/`LimitedFlux` per face, gated by a
  global mesh Péclet number, driven by `WENO3`'s own smoothness indicator (#99)
- `operators::check_cfl` — shared explicit-advection CFL check (FV, WENO, limiters,
  adaptive)

### Changed

- `FDGradientCalculator`/`FDLaplacianCalculator` delegate their stencil math to
  `operators::fd` instead of hand-rolled loops (#94, DD-012)
- `FluxBoundary` drops `Copy`/`PartialEq`/`Eq` (`Arc<dyn BoundaryCondition>` has neither) —
  `FVCenteredFlux`/`FVUpwindFlux`/`WENO3`/`WENO5` drop `Copy` accordingly
- Wide-stencil divergence helpers (`periodic_wide_divergence`, `truncated_wide_divergence`,
  `ghost_padded_field`, `ghost_cell_wide_divergence`, `wrap`) relocated from
  `operators::weno` to `operators` (`pub(crate)`) — shared by `operators::limiters` and
  `operators::adaptive`

### Fixed

- No user-facing bug in this release; two internal design errors were caught and
  corrected before merge (see `DESIGN_DECISIONS.md` for the full account, not silently
  fixed): WENO's initial nodal-semantics claim (DD-041), and `FluxBoundary::GhostCell`'s
  initial single-point `ghost_value` signature being insufficient for WENO's wider
  stencils (DD-042, amendment 1)

## [0.4.0] — 2026-06-25

### Added

- `SteppableSolver` trait in `src/solver/methods/mod.rs` — single-step `step()` primitive,
  extracted from existing `ForwardEulerSolver`/`RK4Solver` logic with no behaviour change
  (Ref #82, DD-031)
- `RK4Solver` — explicit, 4-stage Runge-Kutta, order 4 (#41)
- `MultiDomainOrchestrator` — advances each `Domain` with its own `SteppableSolver` and
  invokes the `Scenario`'s registered `CouplingOperator`s between steps; lets a coupled
  scenario mix integrators per domain (partitioned coupling) (Ref #82, DD-031)
- Integration test for the lahar–lake prototype — first real time loop over coupled domains
  via `MultiDomainOrchestrator`; validates orchestration only, not lahar physics
  (#40, Ref #82, DD-031)
- `LinearSolver` trait + `NalgebraDenseSolver` (dense LU) default implementation (#13, DD-013)
- `finite_difference_jacobian`, `theta_method_step` in `src/solver/methods/implicit.rs` —
  shared by `BackwardEulerSolver` (θ=1) and `CrankNicolsonSolver` (θ=0.5)
  (#43, Refs #13, #86, DD-013, DD-033)
- `BackwardEulerSolver` — implicit, 1st order (#43, Ref #86, DD-033)
- `CrankNicolsonSolver` — semi-implicit, 2nd order (#43, Ref #86, DD-033)
- `SteppableSolver::history_depth()` and a `history: &[ContextValue]` parameter on `step()` —
  supports multi-step integrators; defaults to `0`, no change for one-step solvers
  (Ref #87, DD-034)
- `BDF2Solver` — implicit multi-step, 2nd order; starts with one Backward Euler step when
  history is empty (#44, Ref #87, DD-034)
- `SteppableSolver::solve_fixed_step()` default method — factors out the fixed-step loop
  previously duplicated near-verbatim across `ForwardEulerSolver`, `RK4Solver`,
  `BackwardEulerSolver`, `CrankNicolsonSolver`, `BDF2Solver` (Ref #88, DD-035)
- `StepControl::Adaptive` activated (`dt_init`, `dt_min`, `dt_max`, `rtol`, `atol`) — reserved
  since DD-021, v0.1.0, activated alongside DoPri45 (Ref #89, DD-036)
- `StepSizeController` — PI step-size controller with a source-agnostic error norm, shared
  across adaptive integrators (Ref #89, DD-036)
- `DoPri45Solver` — Dormand-Prince, 7-stage FSAL, adaptive step, order 5; `Solver` only, not
  `SteppableSolver` (#42, Ref #89, DD-036)
- `OperatorSplittingSolver`, `SplitOperator`, `SplittingScheme` in
  `src/solver/methods/imex.rs` — n ≥ 2 operator splitting integrator (Strang scheme); `Solver`
  only, not `SteppableSolver` (#45, DD-037)
- `CompositeModel` in `src/model/composite.rs` — sums multiple `PhysicalModel` contributions
  on the same state; used as the outer `Domain`'s bookkeeping model for IMEX scenarios and as
  a monolithic reference solution (#45, DD-037)
- `IntegratorKind::Imex` (#45, DD-037)

### Changed

- `SteppableSolver::step()` gains a `history: &[ContextValue]` parameter on every
  implementor — no behaviour change for one-step solvers, which ignore it (Ref #87, DD-034)
- `Solver::solve()` on `ForwardEulerSolver`, `RK4Solver`, `BackwardEulerSolver`,
  `CrankNicolsonSolver`, `BDF2Solver` reduced to a single call to `solve_fixed_step()` (Ref #88, DD-035)
- `BoundaryCondition` now requires `Send + Sync` supertraits — additive; existing
  implementations (`DanckwertsInlet`, `DanckwertsOutlet`) already satisfy it trivially.
  Required because `SplitOperator` is the first place a `Domain` is owned rather than
  borrowed by a `Send + Sync`-bound type (`Solver: Send + Sync`) (#45, DD-037)

### Fixed

- `ForwardEulerSolver` — boundary-condition handling correction
- `RK4Solver` — boundary conditions now applied at every Runge-Kutta stage, not once per
  outer step; floating-point accumulation error in the multi-stage state combination

## [0.3.0] — 2026-06-03

### Added

- `PhysicalQuantity` indexed enum in `src/context/quantity.rs` — variants
  `Concentration { component }`, `Temperature`, `Pressure`, `Velocity { component }`,
  `Custom { name: Cow<'static, str>, component }` with idiomatic constructors
  (`concentration()`, `temperature()`, `pressure()`, `velocity()`, `custom(name)`) (DD-010, #10)
- `MultiDomainState` in `src/context/state.rs` — `HashMap<(DomainId, PhysicalQuantity), ContextValue>`
  composite key; explicit entry serialisation for external tool readability (DD-010, #38)
- `Interface` struct in `src/coupling/mod.rs` — source domain, target domain, optional label (DD-011, #11)
- `CouplingOperator` trait in `src/coupling/mod.rs` — `RequiresContext + Send + Sync` supertraits,
  object-safe, INV-3 active from v0.3.0 (DD-011, #11)
- `Scenario::with_coupling()` builder — registers a `CouplingOperator` and its `Interface` (#73)
- `Scenario::couplings()`, `Scenario::interfaces()`, `Scenario::n_couplings()` accessors (#73)
- `Scenario::context_requirements()` now aggregates variables from coupling operators (J3, #73)
- `serde_json` in `[dev-dependencies]` for serde feature tests (#69)
- Integration test `tests/coupling_proto.rs` — proto lahar–lake validates INV-3 structural
  contract (J3 exit criterion, #73)

### Changed

- `src/context/mod.rs` — `quantity` and `state` modules declared and re-exported

### Feature flags

- `serde` — `PhysicalQuantity` and `MultiDomainState` annotated with
  `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` (#69)

## [0.2.0] — 2026-05-02

### Added

- `BoundaryCondition` trait requires `RequiresContext + Debug` supertraits (DD-008)
- `BoundaryLocation` enum with nine variants including `PhaseInterface`, `CouplingInterface`,
  `Custom(Cow<'static, str>)` (DD-008)
- Calculator chain with hybrid two-path ordering: priority fast path + Kahn topological sort (DD-009)
- Built-in calculator suite in `src/context/calculators/`: `TimeCalculator`, `TimeStepCalculator`,
  `FDGradientCalculator`, `FDLaplacianCalculator`, `TrapezoidalIntegral`, `ExternalTabulated` (DD-009)
- `FDScheme` and `Interpolation` enums
- `ContextVariable::External::name` migrated from `&'static str` to `Cow<'static, str>`
- `OxiflowError::PreconditionFailed { context, message }` variant
- `SimulationSnapshot` two-phase restore contract (DD-025)
- Danckwerts inlet feed concentration read from `ComputeContext` via `ContextVariable`
- Integration test `tests/solver_pipeline_chromatography.rs`

## [0.1.0] — 2026-04-12

### Added

- `ContextValue` generic enum: `Scalar`, `Vector`, `Matrix`, `Field2D`, `Boolean`
- `OxiflowError` typed error enum (DD-004)
- `RequiresContext` trait — declarative variable dependency system
- `ComputeContext` — type-safe context access (DD-005, DD-006)
- `Mesh` abstract trait — INV-1 active (DD-003)
- `UniformGrid1D` — structured 1D mesh implementation
- `PhysicalModel` trait — WHAT/HOW separation
- `Scenario` / `SolverConfiguration` / `Solver` architecture
- `DomainId` typed domain identifier
- `ContextVariable` typed key enum

## [0.0.5] — 2026-04-15

### Added

- `Cargo.toml`: full project structure — dependencies (`nalgebra`, `thiserror`),
  features skeleton (`parallel`, `serde`, `hdf5`), crates.io metadata, MSRV 1.80 (#20)
- `src/`: module skeleton — `context`, `mesh`, `model`, `boundary`,
  `solver`, `operators`, `coupling` (#21)
- CI pipeline `ci.yml`: `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test`, `cargo doc` → GitHub Pages (#23)
- `.codecov.yml`: 80% coverage threshold, `default` flag active (#23)
- `CHANGELOG.md`: initial Keep a Changelog entry (#23)

## [0.0.1] — 2026-02-15

### Added

- Reserved `oxiflow` crate name on crates.io
- Apache 2.0 license (`LICENSE`, `NOTICE`)
- Initial `README.md`
- CI infrastructure: GitHub → GitLab mirror (`mirror.yml`, `mirror-patch.yml`),
  coverage (`coverage.yml`)

---

[Unreleased]: https://github.com/biface/oxiflow/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/biface/oxiflow/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/biface/oxiflow/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/biface/oxiflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/biface/oxiflow/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/biface/oxiflow/compare/v0.0.5...v0.1.0
[0.0.5]: https://github.com/biface/oxiflow/compare/v0.0.1...v0.0.5
[0.0.1]: https://github.com/biface/oxiflow/releases/tag/v0.0.1
