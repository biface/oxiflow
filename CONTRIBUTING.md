# Contributing to oxiflow

## Types of contribution

- **Bug fixes** — open an issue first to confirm the bug, then submit a PR.
- **New features** — open an issue and wait for the maintainer's agreement before writing
  code. Features that conflict with the design invariants (see [DEVELOPMENT.md](DEVELOPMENT.md))
  will not be accepted.
- **Documentation and examples** — always welcome, no prior discussion needed.
- **Benchmarks and tests** — always welcome, especially for INV components
  (`mesh`, `coupling`, `solver_spatial`).
- **Niche frameworks** — building a domain-specific `oxiflow-*` crate is explicitly
  encouraged. See the dedicated section below.

## Workflow

```
1. Fork the repository on GitHub
2. Create a branch from develop — not from main
       git checkout develop
       git pull upstream develop
       git checkout -b fix/my-bug        # or feat/, docs/, bench/, chore/
3. Make your changes
4. Open a pull request against develop
```

`main` receives merges from `develop` at each release only.
Direct PRs against `main` will be redirected to `develop`.

## Technical requirements

Every pull request must satisfy all of the following before merging.

**Tests**
- `cargo test --all-features` passes.
- New public functions have at least one test.
- `cargo test --test fem_invariants` passes.

**Code quality**
- `cargo fmt --all` applied, no uncommitted formatting changes.
- `cargo clippy --all-targets --all-features -- -D warnings` clean.

**Coverage**
- Overall coverage ≥ 85% (tracked on Codecov).
- INV components (`mesh`, `coupling`, `solver_spatial`) ≥ 90%.
- If your change drops coverage, add the missing tests before requesting review.

**CHANGELOG**
- Add an entry under `[Unreleased]` in `CHANGELOG.md`.
- Use `### Added`, `### Changed`, `### Fixed`, or `### Breaking`.
- One line per logical change.

## Design invariants

Any contribution touching `src/mesh/`, `src/coupling/`, `src/solver/spatial/`, or any
public trait must preserve the four design invariants:

- **INV-1** — `Mesh` remains abstract; `PhysicalState` must not gain grid assumptions.
- **INV-2** — `DiscreteOperator` remains abstract; integrators stay generic over the scheme.
- **INV-3** — `CouplingOperator` supports distinct domains with moving interfaces.
- **INV-4** — All public traits remain object-safe; no breaking change for external crates.

INV-4 is critical from v2.0 onwards: niche frameworks published by third parties on
crates.io depend on the engine's public API being stable and object-safe. Violations block
merging regardless of test results.

## Building a niche framework

If you are developing a domain-specific framework (`oxiflow-chrom`, `oxiflow-geo`,
or any other domain), here is the recommended approach:

**Before starting**, open a GitHub Discussion describing the domain, the models you plan
to implement, and the target audience. This avoids overlap with ongoing work and allows
the framework to be listed in the official ecosystem.

**Structure of a niche framework crate:**

```
oxiflow-yourdom/
├── Cargo.toml          # depends on oxiflow = "2", separate versioning
├── NOTICE              # required by Apache 2.0 — must mention oxiflow's copyright
├── LICENSE             # your own Apache 2.0 (or compatible) license
├── src/
│   ├── lib.rs
│   ├── models/         # PhysicalModel implementations
│   ├── boundary/       # BoundaryCondition implementations
│   ├── calculators/    # ContextCalculator implementations
│   └── config.rs       # TOML deserialization (optional, for CLI integration)
├── examples/
└── tests/
```

**Registration pattern** — implement the plugin registration function so that
your framework can integrate with the `oxiflow` CLI (v3.0):

```rust
// In oxiflow-yourdom/src/lib.rs
pub fn register(registry: &mut oxiflow::PluginRegistry) {
    registry.register_model("your-model-name", YourModelFactory);
    registry.register_bc("your-bc-name",       YourBCFactory);
}
```

**Naming convention** — use `oxiflow-` as prefix on crates.io. This is not enforced
but makes discovery easier and signals compatibility with the engine.

**Coverage** — aim for ≥ 80% coverage in your framework crate. Use the same
`cargo-llvm-cov` + Codecov setup as the engine if you want a coverage badge.

**Once published**, open a PR against the engine repository to add your crate to the
ecosystem list in `DEVELOPMENT.md`. This is the only contribution to the engine repository
required for a third-party framework.

## Pull request checklist

- [ ] `cargo test --all-features` passes
- [ ] `cargo test --test fem_invariants` passes
- [ ] `cargo fmt --all` applied
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] Codecov ≥ 85% overall and ≥ 90% on INV components
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR targets `develop`, not `main`

## Commit messages

```
type(scope): short description in imperative form

Optional body explaining the why, not the what.
```

Types: `fix`, `feat`, `docs`, `bench`, `test`, `chore`, `refactor`.

## Questions

Open an issue with label `question` or start a GitHub Discussion.

## License

By contributing to oxiflow, you agree that your contributions will be licensed
under the Apache License, Version 2.0, the same license as the project.
The `NOTICE` file must be preserved in any redistribution.
