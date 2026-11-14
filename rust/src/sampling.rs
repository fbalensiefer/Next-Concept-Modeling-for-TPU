//! Categorical and exact speculative decoding primitives.

use crate::{Error, Result};
use rand::{Rng, RngCore};

fn validate(p: &[f64]) -> Result<()> {
    if p.is_empty()
        || p.iter().any(|x| !x.is_finite() || *x < 0.0)
        || (p.iter().sum::<f64>() - 1.0).abs() > 1e-9
    {
        return Err(Error::Probability(
            "probabilities must be finite, nonnegative, and sum to one".into(),
        ));
    }
    Ok(())
}

pub fn categorical<R: Rng + ?Sized>(p: &[f64], rng: &mut R) -> Result<usize> {
    validate(p)?;
    let u = rng.r#gen::<f64>();
    let mut cumulative = 0.0;
    for (i, x) in p.iter().enumerate() {
        cumulative += x;
        if u < cumulative {
            return Ok(i);
        }
    }
    Ok(p.len() - 1)
}

fn residual(p: &[f64], q: &[f64]) -> Result<Vec<f64>> {
    validate(p)?;
    validate(q)?;
    if p.len() != q.len() {
        return Err(Error::Shape("teacher and draft vocabularies differ".into()));
    }
    let mut r: Vec<_> = p.iter().zip(q).map(|(a, b)| (a - b).max(0.0)).collect();
    let z: f64 = r.iter().sum();
    if z > 1e-14 {
        for x in &mut r {
            *x /= z;
        }
    } else {
        r.clone_from_slice(p);
    }
    Ok(r)
}

/// Verify a sequential proposal block. `q[t]` is the exact law used to draw `draft[t]`.
/// `teacher(t, accepted_prefix)` returns p for the current committed context.
pub fn speculative<R, F>(
    draft: &[usize],
    q: &[Vec<f64>],
    rng: &mut R,
    mut teacher: F,
) -> Result<(Vec<usize>, usize)>
where
    R: RngCore + Rng,
    F: FnMut(usize, &[usize]) -> Result<Vec<f64>>,
{
    if draft.len() != q.len() {
        return Err(Error::Shape("draft and proposal lengths differ".into()));
    }
    let mut out = Vec::with_capacity(draft.len() + 1);
    for t in 0..draft.len() {
        let p = teacher(t, &out)?;
        validate(&p)?;
        validate(&q[t])?;
        if draft[t] >= p.len() || q[t].len() != p.len() {
            return Err(Error::Shape("draft token outside vocabulary".into()));
        }
        if rng.r#gen::<f64>() < (p[draft[t]] / q[t][draft[t]]).min(1.0) {
            out.push(draft[t]);
        } else {
            out.push(categorical(&residual(&p, &q[t])?, rng)?);
            return Ok((out, t));
        }
    }
    let bonus = teacher(draft.len(), &out)?;
    out.push(categorical(&bonus, rng)?);
    Ok((out, draft.len()))
}
