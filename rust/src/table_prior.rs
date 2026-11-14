//! Smoothed iid and first-order categorical baselines for host-side studies.
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct TablePrior {
    vocabulary: usize,
    structured: bool,
    alpha: f64,
    marginal: Vec<f64>,
    transitions: BTreeMap<usize, Vec<u64>>,
    totals: Vec<u64>,
}

impl TablePrior {
    pub fn fit(
        rows: &[Vec<usize>],
        vocabulary: usize,
        structured: bool,
        alpha: f64,
    ) -> Result<Self, String> {
        if rows.is_empty()
            || rows[0].is_empty()
            || rows
                .iter()
                .any(|r| r.len() != rows[0].len() || r.iter().any(|x| *x >= vocabulary))
            || vocabulary < 2
            || !alpha.is_finite()
            || alpha <= 0.
        {
            return Err("require nonempty integer rows and valid vocabulary/smoothing".into());
        }
        let mut counts = vec![0u64; vocabulary];
        for &value in rows.iter().flatten() {
            counts[value] += 1;
        }
        let total = rows.iter().map(Vec::len).sum::<usize>() as f64;
        let marginal: Vec<_> = counts
            .iter()
            .map(|n| *n as f64 + alpha / vocabulary as f64)
            .map(|x| x / (total + alpha))
            .collect();
        let mut transitions = BTreeMap::<usize, Vec<u64>>::new();
        if structured {
            for row in rows {
                for pair in row.windows(2) {
                    transitions
                        .entry(pair[0])
                        .or_insert_with(|| vec![0; vocabulary])[pair[1]] += 1;
                }
            }
        }
        let mut totals = vec![0; vocabulary];
        for (context, row) in &transitions {
            totals[*context] = row.iter().sum();
        }
        Ok(Self {
            vocabulary,
            structured,
            alpha,
            marginal,
            transitions,
            totals,
        })
    }
    pub fn marginal(&self) -> &[f64] {
        &self.marginal
    }
    pub fn probability(&self, value: usize, context: Option<usize>) -> Result<f64, String> {
        if value >= self.vocabulary {
            return Err("value outside prior vocabulary".into());
        }
        if let Some(context) = context {
            if context >= self.vocabulary {
                return Err("context outside prior vocabulary".into());
            }
            let total = self.totals[context];
            if self.structured && total > 0 {
                let count = self.transitions.get(&context).map_or(0, |r| r[value]);
                return Ok((count as f64 + self.alpha * self.marginal[value])
                    / (total as f64 + self.alpha));
            }
        }
        Ok(self.marginal[value])
    }
    pub fn nll(&self, rows: &[Vec<usize>]) -> Result<f64, String> {
        if rows.is_empty()
            || rows
                .iter()
                .any(|r| r.is_empty() || r.iter().any(|v| *v >= self.vocabulary))
        {
            return Err("NLL needs nonempty rows within vocabulary".into());
        }
        let mut total = 0.;
        let mut count = 0;
        for row in rows {
            total -= self.marginal[row[0]].ln();
            count += 1;
            if self.structured {
                for pair in row.windows(2) {
                    total -= self.probability(pair[1], Some(pair[0]))?.ln();
                    count += 1;
                }
            } else {
                for &value in &row[1..] {
                    total -= self.marginal[value].ln();
                    count += 1;
                }
            }
        }
        Ok(total / count as f64)
    }
    pub fn sample(
        &self,
        seed: u64,
        count: usize,
        length: usize,
    ) -> Result<Vec<Vec<usize>>, String> {
        if length == 0 {
            return Err("sample length must be positive".into());
        }
        let mut state = seed;
        let mut output = vec![vec![0; length]; count];
        for row in &mut output {
            row[0] = categorical(&self.marginal, unit(&mut state));
            for pos in 1..length {
                let context = row[pos - 1];
                let mut next = categorical(&self.marginal, unit(&mut state));
                let total = self.totals[context];
                if self.structured
                    && total > 0
                    && unit(&mut state) < total as f64 / (total as f64 + self.alpha)
                {
                    let probabilities = self.transitions[&context]
                        .iter()
                        .map(|x| *x as f64 / total as f64)
                        .collect::<Vec<_>>();
                    next = categorical(&probabilities, unit(&mut state));
                }
                row[pos] = next;
            }
        }
        Ok(output)
    }
    /// Numeric payload size only; allocation and map overhead are excluded.
    pub fn storage_bytes(&self) -> usize {
        self.marginal.len() * 16
            + self
                .transitions
                .values()
                .map(|_| self.vocabulary * 8 + 8)
                .sum::<usize>()
    }
}
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
}
fn categorical(probabilities: &[f64], draw: f64) -> usize {
    let mut sum = 0.;
    for (i, p) in probabilities.iter().enumerate() {
        sum += p;
        if draw < sum {
            return i;
        }
    }
    probabilities.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rows_back_off_and_sample_deterministically() {
        let data = vec![vec![0, 1, 0, 1]; 100];
        let prior = TablePrior::fit(&data, 3, true, 1.).unwrap();
        for context in [None, Some(0), Some(1), Some(2)] {
            assert!(
                ((0..3)
                    .map(|v| prior.probability(v, context).unwrap())
                    .sum::<f64>()
                    - 1.)
                    .abs()
                    < 1e-12
            );
        }
        assert_eq!(prior.probability(1, Some(2)).unwrap(), prior.marginal()[1]);
        let samples = prior.sample(12, 1000, 2).unwrap();
        assert_eq!(samples, prior.sample(12, 1000, 2).unwrap());
        assert!(samples.iter().filter(|r| r[0] != r[1]).count() > 980);
        assert!(prior.nll(&data).unwrap() < 0.2);
    }
}
