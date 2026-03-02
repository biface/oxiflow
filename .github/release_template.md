# Release vX.Y.Z — [Title]

> Milestone: [link]

## Changes

### Added
-

### Changed
-

### Fixed
-

### Breaking
-

## FEM Invariants

| Invariant | Status |
|---|---|
| INV-1 Abstract Mesh | ✅ / ⏳ Jn |
| INV-2 DiscreteOperator | ✅ / ⏳ Jn |
| INV-3 CouplingOperator | ✅ / ⏳ Jn |
| INV-4 Plugin-safe API | ✅ / ⏳ Jn / N/A |

## Checklist

- [ ] `cargo test --all-features` passes
- [ ] `cargo test --test fem_invariants` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `Cargo.toml` version updated
- [ ] `CHANGELOG.md` updated
- [ ] Release notes at `.github/release_notes/vX.Y.Z.md`

---
*See the [full CHANGELOG](../../CHANGELOG.md) for detailed history.*
