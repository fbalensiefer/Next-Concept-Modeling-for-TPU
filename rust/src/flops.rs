//! Analytic operation estimates; these are not compiler or hardware counters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlopModel {
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub mlp_ratio: usize,
    pub vocabulary: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CodecFlopModel {
    pub batch: usize,
    pub length: usize,
    pub dim: usize,
    pub vocabulary: usize,
    pub levels: usize,
    pub codes: usize,
}

impl FlopModel {
    pub fn new(width: usize, layers: usize, heads: usize) -> Self {
        Self {
            width,
            layers,
            heads,
            mlp_ratio: 4,
            vocabulary: 256,
        }
    }
}

pub fn transformer_forward(c: FlopModel, batch: usize, length: usize, head: bool) -> f64 {
    let (b, n, w, r, h, v) = (
        batch as f64,
        length as f64,
        c.width as f64,
        c.mlp_ratio as f64,
        c.heads as f64,
        c.vocabulary as f64,
    );
    let attention = (8. + 4. * r) * b * n * w * w + 4. * b * n * n * w;
    let elementwise = 20. * b * n * w + 8. * b * n * r * w + 5. * b * h * n * n;
    let output = if head {
        2. * b * n * w * v + 5. * b * n * v
    } else {
        0.
    };
    c.layers as f64 * (attention + elementwise) + output
}

pub fn ar_training_step(c: FlopModel, batch: usize, length: usize, parameters: usize) -> f64 {
    3. * transformer_forward(c, batch, length, true) + 10. * parameters as f64
}

pub fn ar_cached_decode_step(c: FlopModel, batch: usize, capacity: usize, head: bool) -> f64 {
    let (b, n, w, r, h, v) = (
        batch as f64,
        capacity as f64,
        c.width as f64,
        c.mlp_ratio as f64,
        c.heads as f64,
        c.vocabulary as f64,
    );
    c.layers as f64
        * ((8. + 4. * r) * b * w * w
            + 4. * b * n * w
            + 20. * b * w
            + 8. * b * r * w
            + 5. * b * h * n)
        + if head {
            2. * b * w * v + 5. * b * v
        } else {
            0.
        }
}

pub fn ar_cache_prefill_step(c: FlopModel, batch: usize, capacity: usize) -> f64 {
    let layer = ar_cached_decode_step(c, batch, capacity, false) / c.layers as f64;
    let last_projection =
        6. * batch as f64 * (c.width * c.width) as f64 + 10. * batch as f64 * c.width as f64;
    (c.layers.saturating_sub(1)) as f64 * layer + last_projection
}

pub fn ar_cached_generation(
    c: FlopModel,
    batch: usize,
    capacity: usize,
    prompt_len: usize,
    new_tokens: usize,
) -> f64 {
    prompt_len as f64 * ar_cache_prefill_step(c, batch, capacity)
        + new_tokens as f64 * ar_cached_decode_step(c, batch, capacity, true)
}

pub fn codec_forward(c: CodecFlopModel, encoder: bool, decoder: bool) -> f64 {
    let (b, n, d) = (c.batch as f64, c.length as f64, c.dim as f64);
    let conv = 16. * b * n * d * d + 28. * b * n * d;
    let mut count = 2. * conv * (encoder as u8 as f64) + 2. * conv * (decoder as u8 as f64);
    if encoder {
        count += 2. * b * c.levels as f64 * n * c.codes as f64 * d
            + 4. * c.levels as f64 * b * n * n * d;
    }
    if decoder {
        count += 2. * b * n * d * c.vocabulary as f64 + 5. * b * n * c.vocabulary as f64;
    }
    count
}

pub fn ncm_codec_step(
    batch: usize,
    length: usize,
    dim: usize,
    parameters: usize,
    levels: usize,
    codes: usize,
    vocabulary: usize,
) -> f64 {
    3. * codec_forward(
        CodecFlopModel {
            batch,
            length,
            dim,
            vocabulary,
            levels,
            codes,
        },
        true,
        true,
    ) + 10. * parameters as f64
}

pub fn ncm_prior_forward(
    c: FlopModel,
    batch: usize,
    lengths: &[usize],
    dim: usize,
    prompt_len: usize,
    ar_coarse_levels: usize,
    cached: bool,
) -> f64 {
    let mut count = 0.;
    let max_length = lengths.iter().copied().max().unwrap_or(0);
    for (level, &length) in lengths.iter().enumerate() {
        if cached && level < ar_coarse_levels {
            let capacity = prompt_len + length;
            count += prompt_len as f64 * ar_cache_prefill_step(c, batch, capacity)
                + length as f64 * ar_cached_decode_step(c, batch, capacity, true);
        } else {
            count += transformer_forward(c, batch, prompt_len + length, false)
                + 2. * batch as f64 * length as f64 * c.width as f64 * c.vocabulary as f64
                + 5. * batch as f64 * length as f64 * c.vocabulary as f64;
        }
        count += 2. * batch as f64 * length as f64 * dim as f64 * c.width as f64
            + 4. * batch as f64 * length as f64 * max_length as f64 * dim as f64;
    }
    count
}

#[allow(clippy::too_many_arguments)]
pub fn ncm_prior_step(
    c: FlopModel,
    batch: usize,
    lengths: &[usize],
    dim: usize,
    prompt_len: usize,
    parameters: usize,
    ar_coarse_levels: usize,
    codes: usize,
    decoder_trainable: bool,
    teacher: Option<(FlopModel, usize)>,
) -> f64 {
    let length = lengths.iter().copied().max().unwrap_or(0);
    let codec = CodecFlopModel {
        batch,
        length,
        dim,
        vocabulary: c.vocabulary,
        levels: lengths.len(),
        codes,
    };
    let mut count = 3.
        * ncm_prior_forward(c, batch, lengths, dim, prompt_len, ar_coarse_levels, false)
        + 10. * parameters as f64
        + codec_forward(codec, true, false)
        + (if decoder_trainable { 3. } else { 2. }) * codec_forward(codec, false, true);
    if let Some((teacher_config, teacher_length)) = teacher {
        count += transformer_forward(teacher_config, batch, teacher_length, true);
    }
    count
}

pub fn speculative_round(draft: f64, verify: f64, rounds: f64) -> serde_json::Value {
    serde_json::json!({"analytic_flops_per_round": draft + verify, "measured_rounds": rounds,
        "analytic_flops": (draft + verify) * rounds, "analytic_label": "analytic estimate"})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_estimates_include_training_cost_and_cached_capacity() {
        let c = FlopModel::new(8, 1, 2);
        assert!(ar_training_step(c, 4, 128, 100) > 3. * transformer_forward(c, 4, 128, true));
        for capacity in [8, 64] {
            assert_eq!(
                ar_cache_prefill_step(c, 1, capacity),
                ar_cache_prefill_step(c, 1, 8)
            );
        }
        let codec = CodecFlopModel {
            batch: 1,
            length: 8,
            dim: 4,
            vocabulary: 256,
            levels: 3,
            codes: 4,
        };
        assert!(codec_forward(codec, true, false) > 0.);
        assert!(ncm_prior_step(c, 1, &[1, 2, 4], 4, 0, 100, 1, 4, false, None) > 10. * 100.);
        assert_eq!(speculative_round(100., 200., 3.5)["analytic_flops"], 1050.);
    }
}
