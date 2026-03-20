# Changelog

All notable changes to oxiflow are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
oxiflow adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/biface/oxiflow/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/biface/oxiflow/compare/v0.0.1...v0.0.5
[0.0.1]: https://github.com/biface/oxiflow/releases/tag/v0.0.1
