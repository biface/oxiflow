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

Development is organised in sprints, one per milestone. Every pull request
targets the `update/x.y.z` branch of an open sprint. Sprints follow the
milestones: when the milestone of an issue has no open sprint yet, the work
waits for it.

### Branches and versions

```
update/x.y.z  --PR-->  staging/x.y.z  --PR-->  main  --tag vx.y.z-->  crates.io
```

| Branch | Role | Version in `Cargo.toml` |
|---|---|---|
| `update/x.y.z` | development branch of the sprint | `x.y.z-rc.N` |
| `staging/x.y.z` | copy of `main` that receives the sprint, to check that nothing breaks | `x.y.z` |
| `main` | stable releases only | a release version, no pre-release suffix |

1. Changes are made on `update/x.y.z`, at version `x.y.z-rc.N`.
2. A pull request from `update/x.y.z` to `staging/x.y.z` runs the full test
   suites. Once it is merged, the version is set to `x.y.z` and the code is
   frozen: only `README.*`, `CHANGELOG.md` and similar files may still change.
3. A pull request from `staging/x.y.z` to `main` adds a packaging rehearsal.
4. The tag `vx.y.z`, on `main`, publishes the crate to crates.io. The sprint
   branches are then deleted.

`scripts/check-version.sh` checks the version column, in CI and in the
pre-push hook. A pull request to `staging/x.y.z` must carry an `x.y.z-rc.N`
version; one from `staging/x.y.z` to `main`, exactly `x.y.z`.

### Contributing from a fork

```
1. Fork the repository on GitHub
2. Branch from the update branch of the sprint
       git fetch upstream
       git checkout -b fix/my-bug upstream/update/x.y.z
3. Enable the git hooks (see below) and make your changes
4. Open a pull request against update/x.y.z
```

Branch names are a recommendation: the prefix is the commit type of the change
(`feat/`, `fix/`, `docs/`, `bench/`, `chore/`…), and a `feature/` branch is
just as welcome. Contributor branches are consolidated into `update/x.y.z`
before the sprint goes to `staging/x.y.z`. Contributor pull requests against
`staging/*` or `main` will be redirected.

## Local git hooks

Versioned hooks live in `.githooks/`. Enable them once per clone:

```
git config core.hooksPath .githooks
```

| Hook | What it runs |
|---|---|
| `pre-commit` | refuses staged paths matching `oxiflow.privatePath` (below), then `cargo fmt --check` |
| `commit-msg` | checks the Conventional Commits header (see [Commit messages](#commit-messages)) |
| `pre-push` | for each pushed branch, the `Cargo.toml` version against the branch; for a tag `vX.Y.Z`, the tag against the version and the tagged commit against `<remote>/main`; then, when a branch is pushed, `cargo clippy --locked --all-features --all-targets -- -D warnings`, `cargo test --locked --lib` and `cargo test --locked --doc` |

The pre-push hook is the local counterpart of the checks CI runs on every push.
Its `--all-features` build needs the HDF5 development files (`libhdf5-dev` on
Debian and Ubuntu). Any hook can be bypassed with `--no-verify`
(`git commit --no-verify`, `git push --no-verify`); CI remains the authority.

**Private paths.** Working files that must never reach the repository can be
declared locally, one pattern per entry:

```
git config --add oxiflow.privatePath '*PROJECT_STATUS*'
```

A pattern is an `sh` `case` pattern, matched against the whole path from the
root of the repository; unlike shell globbing, `*` also matches `/`. Deleting
such a file is allowed, so that one committed by mistake can be removed. With
no pattern configured, nothing is checked.

**Limits.**
- `cargo fmt --check` and the cargo commands of the pre-push hook work on the
  working tree, not on the index nor on the commits being pushed. The pre-push
  hook prints a note when tracked files have uncommitted changes.
- A tag is checked against `<remote>/main` as known locally: that reference
  moves on `git fetch` and on a push to `main`. Run `git fetch` first when
  `main` was merged on GitHub, and push tags through a named remote.
- IDEs run the hooks too; in RustRover, keep *Run Git hooks* enabled in the
  commit and push dialogs.

## What CI runs, and when

| Event | Workflows and checks |
|---|---|
| Push to any branch | `Format`, `Version check`, `Clippy` (default features, each feature alone, all features), `Unit and doc tests`, `Security audit` |
| Pull request to `staging/**` or `main` | the above, plus `Test` (one job per feature set), `MSRV`, `SemVer checks`, `Documentation`, `Reference examples`, `Coverage` |
| Pull request to `main` | in addition, `Publish dry run` |
| Push to `main` | `Coverage` (Codecov reference), Release Drafter; the documentation is deployed to GitHub Pages once CI succeeds |
| Tag `vX.Y.Z` | `publish.yml`: `Verify release`, then `Publish to crates.io` |
| Every Monday, 06:00 UTC | `Security audit` |

Pull requests to `update/x.y.z` run none of these checks in this repository:
the push checks run in your fork if GitHub Actions is enabled there, and the full suites
run when the sprint goes to `staging/x.y.z`.

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

**Documentation (local, contributor-facing)**
- The public docs published on GitHub Pages (built by CI, `cargo doc --no-deps
  --all-features`) intentionally only cover the public API surface — internal
  `pub(crate)` items (Newton loop internals, sparse/banded dispatch, etc.) are
  cross-referenced extensively in rustdoc but are not part of that public site.
- To browse the full internal cross-reference graph locally (recommended before
  reviewing a PR that touches `solver::methods` or `operators`):
  ```
  cargo doc --no-deps --all-features --document-private-items --open
  ```
- This command is not run in CI and its output is never published — local use only.

## Design invariants

Any contribution touching `src/mesh/`, `src/coupling/`, `src/solver/spatial/`, or any
public trait must preserve the four design invariants:

- **INV-1** — `Mesh` remains abstract; no public API exposes `dx`, `nx` or raw indices.
- **INV-2** — `DiscreteOperator` remains abstract; integrators stay generic over the scheme.
- **INV-3** — Inter-domain coupling goes through `CouplingOperator` only, never into a solver.
- **INV-4** — All public traits remain object-safe; no breaking change for external crates.

INV-4 is critical from v0.30.0 onwards: niche frameworks published by third parties on
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
- [ ] PR targets the `update/x.y.z` branch of the open sprint

## Commit messages

```
type(scope): short description in imperative form

Optional body explaining the why, not the what.
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
`ci`, `chore`, `revert`, `bench`.

The scope is optional (lowercase letters, digits, `.`, `_`, `/`, `-`). A `!`
before the colon marks a breaking change, as in
`fix(mesh)!: return OxiflowError from UniformGrid1D::new`. Headers generated by
git (`Merge …`, `Revert "…"`, `fixup! …`, `squash! …`, `amend! …`) are
accepted. The `commit-msg` hook checks the header.

## Questions

Open an issue with label `question` or start a GitHub Discussion.

## License

By contributing to oxiflow, you agree that your contributions will be licensed
under the Apache License, Version 2.0, the same license as the project.
The `NOTICE` file must be preserved in any redistribution.
