//! A deterministic residual vector quantizer over fixed codebooks.

use crate::{Array1, Array2, Error, Result};

#[derive(Clone, Debug)]
pub struct ResidualVq {
    pub codebooks: Vec<Array2<f32>>,
}

impl ResidualVq {
    pub fn new(codebooks: Vec<Array2<f32>>) -> Result<Self> {
        let Some(first) = codebooks.first() else {
            return Err(Error::Shape("at least one codebook required".into()));
        };
        let dim = first.ncols();
        if dim == 0 || codebooks.iter().any(|c| c.nrows() == 0 || c.ncols() != dim) {
            return Err(Error::Shape(
                "codebooks must have nonempty rows and a shared nonzero dimension".into(),
            ));
        }
        Ok(Self { codebooks })
    }

    pub fn encode(&self, x: &Array1<f32>) -> Result<Vec<usize>> {
        if x.len() != self.codebooks[0].ncols() {
            return Err(Error::Shape("vector dimension mismatch".into()));
        }
        let mut residual = x.clone();
        let mut ids = Vec::with_capacity(self.codebooks.len());
        for book in &self.codebooks {
            let mut best = (f32::INFINITY, 0usize);
            for (i, row) in book.rows().into_iter().enumerate() {
                let d = residual
                    .iter()
                    .zip(row.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if d < best.0 {
                    best = (d, i);
                }
            }
            ids.push(best.1);
            residual -= &book.row(best.1);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[usize]) -> Result<Array1<f32>> {
        if ids.len() != self.codebooks.len() {
            return Err(Error::Shape("one code index per level required".into()));
        }
        let mut out = Array1::zeros(self.codebooks[0].ncols());
        for (book, &id) in self.codebooks.iter().zip(ids) {
            if id >= book.nrows() {
                return Err(Error::Shape("code index outside codebook".into()));
            }
            out += &book.row(id);
        }
        Ok(out)
    }
}
