//! # Module `solver::gpu`
//!
//! GPU architecture layer (DD-026, issue #73; DD-045, issue #127).
//!
//! ## Scope
//!
//! This module owns exactly two things, both HOW-side (never surfacing into
//! `PhysicalModel`/`Domain`/`Scenario`, per DD-045):
//!
//! 1. **Adapter/device acquisition** ([`GpuContext::new`]) — blocking, via
//!    `pollster`, with an automatic `HighPerformance` adapter preference and
//!    an explicit [`OxiflowError::GpuUnavailable`] on failure (never a silent
//!    CPU fallback — the caller decides).
//! 2. **CPU↔GPU boundary conversion** ([`to_gpu_bytes`]) — reinterprets the
//!    contiguous `f64` payloads already inside [`ContextValue`] (DD-026
//!    INV-GPU-1) as raw bytes via `bytemuck::cast_slice`, without requiring
//!    `ContextValue` itself to implement `bytemuck::Pod` (its enum
//!    discriminant never can). Only the numeric payload is cast, not the
//!    enum wrapper.
//!
//! ## What this module does *not* do
//!
//! No compute shader (WGSL), no buffer upload/dispatch, no `wgpu::Device`
//! usage beyond acquisition. Actual GPU-accelerated computation is future
//! work, not part of #74/DD-045 — this module is the boundary layer those
//! future kernels will sit behind.
//!
//! ## Dispatch model (DD-045)
//!
//! No configurable runtime threshold (contrast with `parallel`/DD-014): the
//! `gpu` feature flag is itself the opt-in. Once compiled with `--features
//! gpu`, [`GpuContext::new`] is always the path taken — there is no
//! CPU/GPU switch left to flip at runtime by this module.

use bytemuck::cast_slice;

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;

// ── GpuContext ──────────────────────────────────────────────────────────────

/// Acquired GPU adapter, device and queue — the entry point of the `gpu`
/// feature.
///
/// Construction is blocking ([`GpuContext::new`]), consistent with
/// `Solver::solve()` already being a single blocking call (DD-045): no
/// async runtime is introduced anywhere else in the crate.
pub struct GpuContext {
    /// Kept for adapter introspection (name, backend, limits) — not read by
    /// this skeleton yet, reserved for the future dispatch layer.
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    /// Logical GPU device — will own future compute pipelines/shaders.
    pub device: wgpu::Device,
    /// Command queue — will submit future compute dispatches.
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Acquires a GPU adapter and device, blocking (DD-045).
    ///
    /// Requests a `HighPerformance` adapter (compute workload, not
    /// interactive rendering — no `compatible_surface`). Returns
    /// [`OxiflowError::GpuUnavailable`] if no compatible adapter or device
    /// is found — never a silent CPU fallback; the caller chooses its own
    /// CPU solver in that case.
    ///
    /// # Errors
    ///
    /// Returns [`OxiflowError::GpuUnavailable`] if adapter or device
    /// acquisition fails.
    pub fn new() -> Result<Self, OxiflowError> {
        // `wgpu 26.0.1` API: `InstanceDescriptor` still implements `Default`
        // at this version (removed in a later release, v29+) — explicit
        // backends restrict to the set decided in DD-045 (vulkan/metal/dx12,
        // no gles/webgpu), matching the `features` selected in `Cargo.toml`.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|_| OxiflowError::GpuUnavailable)?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("oxiflow-gpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|_| OxiflowError::GpuUnavailable)?;

        Ok(Self {
            adapter,
            device,
            queue,
        })
    }
}

// ── CPU↔GPU boundary conversion ──────────────────────────────────────────────

/// Reinterprets a numeric [`ContextValue`] payload as raw bytes, ready for
/// GPU buffer upload (DD-026 INV-GPU-5).
///
/// Only the contiguous `f64` payload is cast (`bytemuck::cast_slice`) — never
/// `ContextValue` itself, whose enum discriminant is not `Pod`-compatible.
/// `Boolean` is not a numeric payload and has no meaningful GPU-upload
/// representation here; it returns [`OxiflowError::TypeMismatch`].
///
/// # Errors
///
/// Returns [`OxiflowError::TypeMismatch`] for [`ContextValue::Boolean`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "gpu")]
/// # {
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::solver::gpu::to_gpu_bytes;
/// use nalgebra::DVector;
///
/// let field = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
/// let bytes = to_gpu_bytes(&field).unwrap();
/// assert_eq!(bytes.len(), 3 * std::mem::size_of::<f64>());
/// # }
/// ```
pub fn to_gpu_bytes(value: &ContextValue) -> Result<&[u8], OxiflowError> {
    match value {
        ContextValue::Scalar(s) => Ok(bytemuck::bytes_of(s)),
        ContextValue::Vector(v) | ContextValue::ScalarField(v) => Ok(cast_slice(v.as_slice())),
        ContextValue::Matrix(m) | ContextValue::VectorField(m) => Ok(cast_slice(m.as_slice())),
        ContextValue::Boolean(_) => Err(OxiflowError::TypeMismatch {
            expected: "numeric ContextValue (Scalar/Vector/Matrix/ScalarField/VectorField)",
            actual: "Boolean",
        }),
        // `#[non_exhaustive]` on ContextValue: reserved J7 variants
        // (Tensor4/TensorField) are not yet defined in the enum itself, so
        // this arm is unreachable today, not a gap.
        #[allow(unreachable_patterns)]
        _ => Err(OxiflowError::TypeMismatch {
            expected: "numeric ContextValue (Scalar/Vector/Matrix/ScalarField/VectorField)",
            actual: value.variant_name(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn to_gpu_bytes_scalar_field() {
        let field = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let bytes = to_gpu_bytes(&field).unwrap();
        assert_eq!(bytes.len(), 3 * std::mem::size_of::<f64>());
    }

    #[test]
    fn to_gpu_bytes_vector_field() {
        let field = ContextValue::VectorField(DMatrix::from_element(2, 3, 1.5));
        let bytes = to_gpu_bytes(&field).unwrap();
        assert_eq!(bytes.len(), 6 * std::mem::size_of::<f64>());
    }

    #[test]
    fn to_gpu_bytes_scalar() {
        let value = ContextValue::Scalar(42.0);
        let bytes = to_gpu_bytes(&value).unwrap();
        assert_eq!(bytes.len(), std::mem::size_of::<f64>());
    }

    #[test]
    fn to_gpu_bytes_rejects_boolean() {
        let value = ContextValue::Boolean(true);
        let err = to_gpu_bytes(&value).unwrap_err();
        assert!(matches!(err, OxiflowError::TypeMismatch { .. }));
    }

    /// No GPU hardware in CI (documented limitation, DD-045 consequences) —
    /// `#[ignore]`, run manually on a machine with a real adapter.
    #[test]
    #[ignore = "requires real GPU hardware, not available in CI"]
    fn gpu_context_acquires_adapter() {
        GpuContext::new().unwrap();
    }
}
