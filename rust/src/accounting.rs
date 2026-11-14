//! Likelihood accounting with explicit continuation-byte denominators.

use crate::{Error, Result};

/// Convert total negative log likelihood in nats into bits per byte.
pub fn bits_per_byte(nll_nats: f64, continuation_bytes: usize) -> Result<f64> {
    if !nll_nats.is_finite() || nll_nats < 0.0 || continuation_bytes == 0 {
        return Err(Error::Shape(
            "NLL must be finite/nonnegative and byte count positive".into(),
        ));
    }
    Ok(nll_nats / std::f64::consts::LN_2 / continuation_bytes as f64)
}

/// Convert a likelihood lower bound into a bits-per-byte upper bound.
pub fn bpb_upper_bound(nll_lower_bound_nats: f64, continuation_bytes: usize) -> Result<f64> {
    bits_per_byte(nll_lower_bound_nats, continuation_bytes)
}
