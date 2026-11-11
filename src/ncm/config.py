"""Immutable configuration: changes deliberately create new executables."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RVQConfig:
    lengths: tuple[int, ...]
    codebook_size: int
    dim: int
    decay: float = 0.99
    epsilon: float = 1e-5
    entropy_temperature: float = 1.0
    nested_dropout: float = 0.0

    def __post_init__(self):
        if not self.lengths or any(n <= 0 for n in self.lengths):
            raise ValueError("lengths must be nonempty and positive")
        if tuple(sorted(self.lengths)) != self.lengths:
            raise ValueError("lengths must be ordered coarse to fine")
        if self.codebook_size < 2 or self.dim <= 0:
            raise ValueError("codebook_size >= 2 and dim > 0 required")
        if not 0 <= self.decay < 1 or self.epsilon <= 0:
            raise ValueError("decay must be in [0, 1), epsilon > 0")
        if self.entropy_temperature <= 0 or not 0 <= self.nested_dropout <= 1:
            raise ValueError("invalid temperature or nested_dropout")

    @property
    def levels(self):
        return len(self.lengths)

    @property
    def max_length(self):
        return self.lengths[-1]


@dataclass(frozen=True)
class TransformerConfig:
    width: int = 256
    heads: int = 8
    layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.width, self.heads, self.layers, self.mlp_ratio) <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("width must divide heads; dropout must be in [0, 1)")


@dataclass(frozen=True)
class LossConfig:
    beta: float = 0.25
    entropy: float = 0.01
    concept: float = 0.0
    prior: float = 1.0
    # Empty means linearly decreasing K, ..., 1 (coarse levels get more weight).
    entropy_weights: tuple[float, ...] = ()
