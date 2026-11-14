use ncm::{
    corpus::{CorpusError, document_byte_windows, validate_disjoint_splits},
    evaluation::reconstruction_metrics,
};
use std::{fs, path::PathBuf};

fn fixture(name: &str, text: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("ncm-{name}-{}", std::process::id()));
    fs::write(&p, text).unwrap();
    p
}

#[test]
fn windows_keep_document_boundaries_hash_input_and_count_tails() {
    let p = fixture(
        "windows",
        r#"{"id":"a","text":"abcde"}
{"id":"b","text":"XY"}
"#,
    );
    let w = document_byte_windows(&p, 3).unwrap();
    assert_eq!(w.rows, vec![b"abc".to_vec()]);
    assert_eq!(w.document_ids, vec!["a"]);
    assert_eq!(w.dropped_tail_bytes, 4);
    assert_eq!(
        w.sha256,
        "4c4466f4a2950d941f017cb8cb0528e38b16b76143e3eda95eb8181bcf4558fc"
    );
    fs::remove_file(p).unwrap();
}

#[test]
fn splits_reject_duplicate_id_or_text() {
    let a = fixture("split-a", r#"{"id":"a","text":"same"}"#);
    let b = fixture("split-b", r#"{"id":"b","text":"same"}"#);
    assert!(matches!(
        validate_disjoint_splits(&[&a, &b]),
        Err(CorpusError::Invalid(_))
    ));
    fs::remove_file(a).unwrap();
    fs::remove_file(b).unwrap();
}

#[test]
fn reconstruction_metrics_preserve_mask_and_empty_slices() {
    let m = reconstruction_metrics(
        &[b"a1".to_vec(), b"!?".to_vec()],
        &[b"a0".to_vec(), b"!!".to_vec()],
        &[vec![true, true], vec![true, false]],
    )
    .unwrap();
    assert_eq!(m.valid_bytes, 3);
    assert_eq!(m.errors_by_position, vec![0, 1]);
    assert_eq!(m.digits.accuracy, Some(0.));
    assert_eq!(m.punctuation.bytes, 1);
    assert_eq!(m.punctuation.accuracy, Some(1.));
}
