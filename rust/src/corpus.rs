//! Deterministic byte-corpus loading and document-separated windows.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CorpusError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid JSONL: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid corpus: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Document {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteWindows {
    pub rows: Vec<Vec<u8>>,
    pub document_ids: Vec<String>,
    pub sha256: String,
    pub dropped_tail_bytes: usize,
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn read_documents(path: impl AsRef<Path>) -> Result<Vec<Document>, CorpusError> {
    let raw = fs::read(path)?;
    let text = std::str::from_utf8(&raw).map_err(|e| CorpusError::Invalid(e.to_string()))?;
    let mut docs = Vec::new();
    let mut ids = HashSet::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let doc: Document = serde_json::from_str(line)?;
        if doc.id.is_empty() || doc.text.is_empty() || !ids.insert(doc.id.clone()) {
            return Err(CorpusError::Invalid(
                "documents require unique nonempty IDs and text".into(),
            ));
        }
        docs.push(doc);
    }
    if docs.is_empty() {
        return Err(CorpusError::Invalid("empty corpus".into()));
    }
    Ok(docs)
}

pub fn validate_disjoint_splits(paths: &[impl AsRef<Path>]) -> Result<(), CorpusError> {
    let (mut ids, mut contents) = (HashSet::new(), HashSet::new());
    for path in paths {
        let docs = read_documents(path)?;
        for d in docs {
            if !ids.insert(d.id.clone()) || !contents.insert(sha256(d.text.as_bytes())) {
                return Err(CorpusError::Invalid(
                    "document IDs or contents overlap between splits".into(),
                ));
            }
        }
    }
    Ok(())
}

pub fn document_byte_windows(
    path: impl AsRef<Path>,
    length: usize,
) -> Result<ByteWindows, CorpusError> {
    if length == 0 {
        return Err(CorpusError::Invalid(
            "window length must be positive".into(),
        ));
    }
    let path = path.as_ref();
    let raw = fs::read(path)?;
    let docs = read_documents(path)?;
    let (mut rows, mut owners, mut dropped) = (Vec::new(), Vec::new(), 0);
    for d in docs {
        let bytes = d.text.as_bytes();
        let complete = bytes.len() / length;
        dropped += bytes.len() % length;
        for i in 0..complete {
            rows.push(bytes[i * length..(i + 1) * length].to_vec());
            owners.push(d.id.clone());
        }
    }
    if rows.is_empty() {
        return Err(CorpusError::Invalid(
            "no complete windows at requested length".into(),
        ));
    }
    Ok(ByteWindows {
        rows,
        document_ids: owners,
        sha256: sha256(&raw),
        dropped_tail_bytes: dropped,
    })
}
