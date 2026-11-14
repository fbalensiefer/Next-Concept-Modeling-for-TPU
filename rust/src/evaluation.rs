//! Host-side byte reconstruction summaries.
use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SliceAccuracy {
    pub bytes: usize,
    pub accuracy: Option<f64>,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ReconstructionMetrics {
    pub byte_accuracy: f64,
    pub exact_window_accuracy: f64,
    pub windows: usize,
    pub valid_bytes: usize,
    pub errors_by_position: Vec<usize>,
    pub valid_bytes_by_position: Vec<usize>,
    pub digits: SliceAccuracy,
    pub punctuation: SliceAccuracy,
}

pub fn reconstruction_metrics(
    target: &[Vec<u8>],
    prediction: &[Vec<u8>],
    mask: &[Vec<bool>],
) -> Result<ReconstructionMetrics, String> {
    if target.is_empty() || target.len() != prediction.len() || target.len() != mask.len() {
        return Err("expected matching nonempty window batches".into());
    }
    let width = target[0].len();
    if width == 0
        || target
            .iter()
            .chain(prediction)
            .chain(mask.iter().map(|_| target.first().unwrap()))
            .any(|r| r.len() != width)
        || mask
            .iter()
            .any(|r| r.len() != width || !r.iter().any(|&x| x))
    {
        return Err("window shapes must match and contain valid bytes".into());
    }
    let (mut errors, mut counts) = (vec![0; width], vec![0; width]);
    let (mut hits, mut total, mut exact) = (0, 0, 0);
    let (mut digit_hit, mut digit_n, mut punct_hit, mut punct_n) = (0, 0, 0, 0);
    for ((t, p), m) in target.iter().zip(prediction).zip(mask) {
        let mut row_exact = true;
        for i in 0..width {
            if !m[i] {
                continue;
            }
            total += 1;
            counts[i] += 1;
            let hit = t[i] == p[i];
            if hit {
                hits += 1
            } else {
                errors[i] += 1;
                row_exact = false
            }
            if t[i].is_ascii_digit() {
                digit_n += 1;
                if hit {
                    digit_hit += 1
                }
            }
            if t[i].is_ascii_punctuation() {
                punct_n += 1;
                if hit {
                    punct_hit += 1
                }
            }
        }
        if row_exact {
            exact += 1
        }
    }
    let slice = |n, h| SliceAccuracy {
        bytes: n,
        accuracy: if n > 0 {
            Some(h as f64 / n as f64)
        } else {
            None
        },
    };
    Ok(ReconstructionMetrics {
        byte_accuracy: hits as f64 / total as f64,
        exact_window_accuracy: exact as f64 / target.len() as f64,
        windows: target.len(),
        valid_bytes: total,
        errors_by_position: errors,
        valid_bytes_by_position: counts,
        digits: slice(digit_n, digit_hit),
        punctuation: slice(punct_n, punct_hit),
    })
}
