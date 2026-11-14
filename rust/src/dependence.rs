//! Exact finite-domain dependence calculations with a strict enumeration cap.
use std::collections::HashMap;

pub const DEFAULT_MAX_WORDS: usize = 4096;
pub fn codewords(
    lengths: &[usize],
    vocabulary: usize,
    max_words: usize,
) -> Result<Vec<Vec<usize>>, String> {
    let slots = lengths.iter().sum::<usize>();
    let domain = vocabulary
        .checked_pow(slots as u32)
        .ok_or("code domain exceeds enumeration budget")?;
    if vocabulary < 2 || slots == 0 || lengths.contains(&0) || domain > max_words {
        return Err("Code domain exceeds exact enumeration budget".into());
    }
    Ok((0..domain)
        .map(|index| decode(index, slots, vocabulary))
        .collect())
}
fn decode(mut index: usize, slots: usize, vocab: usize) -> Vec<usize> {
    let mut word = vec![0; slots];
    for i in (0..slots).rev() {
        word[i] = index % vocab;
        index /= vocab;
    }
    word
}
fn checked_joint(joint: &[f64], lengths: &[usize], vocabulary: usize) -> Result<usize, String> {
    let slots: usize = lengths.iter().sum();
    let n = vocabulary
        .checked_pow(slots as u32)
        .ok_or("code domain exceeds exact enumeration budget")?;
    if vocabulary < 2
        || slots == 0
        || lengths.contains(&0)
        || n > DEFAULT_MAX_WORDS
        || joint.len() != n
        || joint.iter().any(|x| !x.is_finite() || *x < 0.)
    {
        return Err("invalid joint distribution or exact enumeration budget exceeded".into());
    }
    Ok(slots)
}
fn entropy(probabilities: impl IntoIterator<Item = f64>) -> f64 {
    probabilities
        .into_iter()
        .filter(|p| *p > 0.)
        .map(|p| -p * p.ln())
        .sum()
}
fn marginal_entropy(joint: &[f64], lengths: &[usize], vocab: usize, keep: &[usize]) -> f64 {
    let words = joint.len();
    let mut marginal = HashMap::<Vec<usize>, f64>::new();
    for (i, p) in joint.iter().enumerate() {
        if *p == 0. {
            continue;
        }
        let word = decode(i, lengths.iter().sum(), vocab);
        let key = keep.iter().map(|slot| word[*slot]).collect::<Vec<_>>();
        *marginal.entry(key).or_default() += p;
    }
    let _ = words;
    entropy(marginal.into_values())
}
pub fn conditional_tc(
    joint: &[f64],
    lengths: &[usize],
    vocabulary: usize,
) -> Result<Vec<f64>, String> {
    let slots = checked_joint(joint, lengths, vocabulary)?;
    let mut start = 0;
    let mut output = Vec::new();
    for &count in lengths {
        let prefix = (0..start).collect::<Vec<_>>();
        let mut value = 0.;
        for slot in start..start + count {
            let mut keep = prefix.clone();
            keep.push(slot);
            value += marginal_entropy(joint, lengths, vocabulary, &keep);
        }
        value -= marginal_entropy(
            joint,
            lengths,
            vocabulary,
            &(0..start + count).collect::<Vec<_>>(),
        );
        value -= (count - 1) as f64 * marginal_entropy(joint, lengths, vocabulary, &prefix);
        output.push(value);
        start += count;
    }
    debug_assert_eq!(start, slots);
    Ok(output)
}
pub fn hard_joint(
    codes: &[Vec<usize>],
    weights: &[f64],
    lengths: &[usize],
    vocabulary: usize,
) -> Result<Vec<f64>, String> {
    let slots: usize = lengths.iter().sum();
    let n = vocabulary
        .checked_pow(slots as u32)
        .ok_or("code domain exceeds budget")?;
    if codes.is_empty()
        || codes.len() != weights.len()
        || n > DEFAULT_MAX_WORDS
        || codes
            .iter()
            .any(|r| r.len() != slots || r.iter().any(|x| *x >= vocabulary))
        || weights.iter().any(|x| !x.is_finite() || *x < 0.)
    {
        return Err("invalid codes, weights, or enumeration budget".into());
    }
    let total: f64 = weights.iter().sum();
    if total <= 0. {
        return Err("weights must have positive sum".into());
    }
    let mut joint = vec![0.; n];
    for (row, w) in codes.iter().zip(weights) {
        let index = row.iter().fold(0, |acc, x| acc * vocabulary + x);
        joint[index] += w / total;
    }
    Ok(joint)
}
pub fn soft_joint(
    probabilities: &[Vec<Vec<f64>>],
    weights: &[f64],
    lengths: &[usize],
    vocabulary: usize,
) -> Result<Vec<f64>, String> {
    let slots: usize = lengths.iter().sum();
    let n = vocabulary
        .checked_pow(slots as u32)
        .ok_or("code domain exceeds budget")?;
    if probabilities.is_empty()
        || probabilities.len() != weights.len()
        || n > DEFAULT_MAX_WORDS
        || weights.iter().any(|x| !x.is_finite() || *x < 0.)
        || probabilities.iter().any(|row| {
            row.len() != slots
                || row
                    .iter()
                    .any(|p| p.len() != vocabulary || p.iter().any(|x| !x.is_finite() || *x < 0.))
        })
    {
        return Err("invalid probabilities, weights, or enumeration budget".into());
    }
    let total: f64 = weights.iter().sum();
    if total <= 0. {
        return Err("weights must have positive sum".into());
    }
    let words = codewords(lengths, vocabulary, DEFAULT_MAX_WORDS)?;
    let mut joint = vec![0.; n];
    for (row, w) in probabilities.iter().zip(weights) {
        for (index, word) in words.iter().enumerate() {
            let mass = word
                .iter()
                .enumerate()
                .fold(1., |m, (slot, value)| m * row[slot][*value]);
            joint[index] += w / total * mass;
        }
    }
    Ok(joint)
}
pub fn direct_factorization_kl(
    joint: &[f64],
    lengths: &[usize],
    vocabulary: usize,
) -> Result<f64, String> {
    let slots = checked_joint(joint, lengths, vocabulary)?;
    let words = codewords(lengths, vocabulary, DEFAULT_MAX_WORDS)?;
    let mut kl = 0.;
    for (index, p) in joint.iter().enumerate() {
        if *p == 0. {
            continue;
        }
        let word = &words[index];
        let mut q = 1.;
        let mut offset = 0;
        for &count in lengths {
            let denominator: f64 = joint
                .iter()
                .zip(&words)
                .filter(|(_, candidate)| candidate[..offset] == word[..offset])
                .map(|(mass, _)| mass)
                .sum();
            for slot in offset..offset + count {
                let numerator: f64 = joint
                    .iter()
                    .zip(&words)
                    .filter(|(_, candidate)| {
                        candidate[..offset] == word[..offset] && candidate[slot] == word[slot]
                    })
                    .map(|(mass, _)| mass)
                    .sum();
                q *= if denominator > 0. {
                    numerator / denominator
                } else {
                    1. / vocabulary as f64
                };
            }
            offset += count;
        }
        kl += p * (p / q.max(1e-300)).ln();
    }
    let _ = slots;
    Ok(kl)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dependence_identities_and_soft_hard_agree() {
        let codes = vec![vec![0, 0, 1], vec![1, 1, 0], vec![1, 0, 1]];
        let weights = [0.2, 0.3, 0.5];
        let hard = hard_joint(&codes, &weights, &[1, 2], 2).unwrap();
        let probabilities = codes
            .iter()
            .map(|r| {
                r.iter()
                    .map(|x| vec![if *x == 0 { 1. } else { 0. }, if *x == 1 { 1. } else { 0. }])
                    .collect()
            })
            .collect::<Vec<Vec<Vec<f64>>>>();
        let soft = soft_joint(&probabilities, &weights, &[1, 2], 2).unwrap();
        assert_eq!(hard, soft);
        let shared = vec![0.5, 0., 0., 0.5];
        assert!((conditional_tc(&shared, &[2], 2).unwrap()[0] - 2f64.ln()).abs() < 1e-12);
        assert!((direct_factorization_kl(&shared, &[2], 2).unwrap() - 2f64.ln()).abs() < 1e-12);
        assert!(codewords(&[20], 2, 4096).is_err());
    }
}
