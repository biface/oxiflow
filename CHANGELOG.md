# Changelog

All notable changes to oxiflow are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
oxiflow adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/biface/oxiflow/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/biface/oxiflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/biface/oxiflow/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/biface/oxiflow/compare/v0.0.5...v0.1.0
[0.0.5]: https://github.com/biface/oxiflow/compare/v0.0.1...v0.0.5
[0.0.1]: https://github.com/biface/oxiflow/releases/tag/v0.0.1
