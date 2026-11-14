//! Host-side token accounting, latency matching, and speculative counters.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub fn continuation_mask(mask: &[Vec<bool>], prompt_len: usize) -> Result<Vec<Vec<bool>>, String> {
    if mask
        .iter()
        .any(|row| row.len() != mask.first().map_or(0, Vec::len))
    {
        return Err("mask rows must have equal lengths".into());
    }
    Ok(mask
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, valid)| *valid && i >= prompt_len)
                .collect()
        })
        .collect())
}

pub fn bpb_from_sums(
    nll_sum: f64,
    mask: &[Vec<bool>],
    prompt_len: usize,
) -> Result<(f64, usize), String> {
    let selected = continuation_mask(mask, prompt_len)?;
    let count = selected.iter().flatten().filter(|valid| **valid).count();
    Ok((
        nll_sum / count.max(1) as f64 / std::f64::consts::LN_2,
        count,
    ))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyPoint {
    pub bpb: f64,
    pub median_ms: f64,
    pub bpb_kind: String,
}

pub fn matched_latency_rows(
    curves: &std::collections::BTreeMap<String, Vec<LatencyPoint>>,
) -> Vec<Value> {
    let names: Vec<_> = curves.keys().collect();
    let mut rows = Vec::new();
    for i in 0..names.len() {
        for j in i + 1..names.len() {
            let (a, b) = (names[i], names[j]);
            let (ca, cb) = (&curves[a], &curves[b]);
            if ca.is_empty() || cb.is_empty() {
                continue;
            }
            let low = ca
                .iter()
                .map(|x| x.bpb)
                .fold(f64::INFINITY, f64::min)
                .max(cb.iter().map(|x| x.bpb).fold(f64::INFINITY, f64::min));
            let high = ca
                .iter()
                .map(|x| x.bpb)
                .fold(f64::NEG_INFINITY, f64::max)
                .min(cb.iter().map(|x| x.bpb).fold(f64::NEG_INFINITY, f64::max));
            let comparable = ca.iter().chain(cb).all(|x| x.bpb_kind == "exact");
            if low > high {
                rows.push(json!({"arm_a": a,"arm_b": b,"status":"no_overlap",
                "bpb_kind_a":ca[0].bpb_kind,"bpb_kind_b":cb[0].bpb_kind}));
                continue;
            }
            let common = (low + high) / 2.;
            let interp = |curve: &[LatencyPoint]| interpolate(curve, common);
            rows.push(json!({"arm_a":a,"arm_b":b,
            "status": if comparable {"overlap"} else {"upper_bound_not_quality_match"},
            "common_bpb":common,"latency_a_ms":interp(ca),"latency_b_ms":interp(cb),
            "bpb_kind_a":ca[0].bpb_kind,"bpb_kind_b":cb[0].bpb_kind,
            "quality_matched_claim_permitted":comparable}));
        }
    }
    rows
}

fn interpolate(curve: &[LatencyPoint], target: f64) -> f64 {
    let mut points: Vec<_> = curve.iter().map(|p| (p.bpb, p.median_ms)).collect();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    if target <= points[0].0 {
        return points[0].1;
    }
    if target >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }
    for pair in points.windows(2) {
        if target <= pair[1].0 {
            let (x0, y0) = pair[0];
            let (x1, y1) = pair[1];
            return y0 + (target - x0) / (x1 - x0) * (y1 - y0);
        }
    }
    points[points.len() - 1].1
}

#[derive(Clone, Debug)]
pub struct SpeculativeBatchStats {
    pub teacher_forward_passes: Vec<usize>,
    pub round_prefix_lengths: Vec<Vec<i64>>,
    pub accepted_by_position: Vec<Vec<u64>>,
    pub proposed_by_position: Vec<Vec<u64>>,
}

pub fn speculative_batch_observation(
    stats: &SpeculativeBatchStats,
    prompt_len: usize,
    new_tokens: usize,
) -> Result<Value, String> {
    if stats.teacher_forward_passes.is_empty()
        || stats.round_prefix_lengths.len() != stats.teacher_forward_passes.len()
    {
        return Err("speculative stats require matching nonempty batch rows".into());
    }
    let rounds = *stats.teacher_forward_passes.iter().max().unwrap();
    if stats
        .round_prefix_lengths
        .iter()
        .any(|row| row.len() < rounds)
    {
        return Err("prefix trace shorter than batch rounds".into());
    }
    let positions: usize = (0..rounds)
        .map(|r| {
            stats
                .round_prefix_lengths
                .iter()
                .enumerate()
                .map(|(b, row)| {
                    if r < stats.teacher_forward_passes[b] {
                        row[r].max(0) as usize
                    } else {
                        prompt_len + new_tokens
                    }
                })
                .max()
                .unwrap_or(0)
        })
        .sum();
    let sum = |rows: &[Vec<u64>]| -> Vec<u64> {
        let width = rows.iter().map(Vec::len).max().unwrap_or(0);
        (0..width)
            .map(|i| rows.iter().map(|r| r.get(i).copied().unwrap_or(0)).sum())
            .collect()
    };
    let accepted = sum(&stats.accepted_by_position);
    let proposed = sum(&stats.proposed_by_position);
    let prefix_sum: usize = stats
        .round_prefix_lengths
        .iter()
        .flatten()
        .filter(|x| **x > 0)
        .map(|x| *x as usize)
        .sum();
    let sequence_rounds: usize = stats.teacher_forward_passes.iter().sum();
    Ok(
        json!({"mean_rounds":sequence_rounds as f64/stats.teacher_forward_passes.len() as f64,
        "batch_rounds":rounds,"prefill_iterations":positions,"accepted":accepted,
        "proposed":proposed,"prefix_sum":prefix_sum,"sequence_rounds":sequence_rounds}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuation_accounting_and_bound_status_match_contract() {
        let mask = vec![vec![true; 6], vec![true, true, true, true, false, false]];
        assert_eq!(bpb_from_sums(12., &mask, 2).unwrap().1, 6);
        let mut curves = std::collections::BTreeMap::new();
        curves.insert(
            "a".into(),
            vec![
                LatencyPoint {
                    bpb: 2.,
                    median_ms: 1.,
                    bpb_kind: "exact".into(),
                },
                LatencyPoint {
                    bpb: 3.,
                    median_ms: 2.,
                    bpb_kind: "exact".into(),
                },
            ],
        );
        curves.insert(
            "b".into(),
            vec![LatencyPoint {
                bpb: 4.,
                median_ms: 1.,
                bpb_kind: "exact".into(),
            }],
        );
        assert_eq!(matched_latency_rows(&curves)[0]["status"], "no_overlap");
    }

    #[test]
    fn speculative_batch_counter_charges_finished_lanes() {
        let stats = SpeculativeBatchStats {
            teacher_forward_passes: vec![2, 4],
            round_prefix_lengths: vec![vec![16, 20, -1, -1], vec![16, 18, 21, 23]],
            accepted_by_position: vec![vec![1, 0], vec![2, 1]],
            proposed_by_position: vec![vec![2, 1], vec![4, 3]],
        };
        let result = speculative_batch_observation(&stats, 16, 10).unwrap();
        assert_eq!(result["prefill_iterations"], 88);
        assert_eq!(result["sequence_rounds"], 6);
        assert_eq!(result["accepted"], json!([3, 1]));
    }
}
