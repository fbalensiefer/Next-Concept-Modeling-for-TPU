//! Native Rust primitives for Next-Concept Modeling.
//!
//! This crate currently ports deterministic byte handling, residual vector
//! quantization, causal categorical sampling, speculative verification, and
//! byte-likelihood accounting. Neural-network training and accelerator support
//! remain in the Python/JAX implementation while those components are ported.

pub mod accounting;
pub mod rvq;
pub mod sampling;
pub mod text;

pub use ndarray::{Array1, Array2};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid shape or empty input: {0}")]
    Shape(String),
    #[error("invalid probability distribution: {0}")]
    Probability(String),
}

pub type Result<T> = std::result::Result<T, Error>;
