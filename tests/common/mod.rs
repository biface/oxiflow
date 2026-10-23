//! # Shared fixtures for the integration tests
//!
//! Cargo compiles each file of `tests/` as its own crate; a file uses these
//! fixtures with `mod common;`. A subdirectory with a `mod.rs` is not itself
//! compiled as a test crate.

pub mod meshes;
