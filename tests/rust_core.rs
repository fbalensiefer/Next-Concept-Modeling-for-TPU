use ncm::{accounting::bits_per_byte, rvq::ResidualVq, sampling::speculative, text};
use ndarray::array;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn byte_round_trip_and_bpb() {
    let bytes = text::encode("NCM λ");
    assert_eq!(text::decode(&bytes).unwrap(), "NCM λ");
    assert_eq!(text::vocabulary_size(), 256);
    assert!((bits_per_byte(std::f64::consts::LN_2 * 8.0, 1).unwrap() - 8.0).abs() < 1e-12);
}

#[test]
fn rvq_encodes_and_decodes_residuals() {
    let q = ResidualVq::new(vec![array![[0., 0.], [1., 0.]], array![[0., 0.], [0., 1.]]]).unwrap();
    let x = array![1.1, 0.9];
    let ids = q.encode(&x).unwrap();
    assert_eq!(ids, vec![1, 1]);
    let y = q.decode(&ids).unwrap();
    assert_eq!(y, array![1., 1.]);
}

#[test]
fn exact_speculative_rejection_returns_normalized_residual() {
    let p = vec![1.0, 0.0];
    let q = vec![0.2, 0.8];
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let (tokens, accepted) = speculative(&[1], &[q], &mut rng, |_, _| Ok(p.clone())).unwrap();
    assert_eq!(accepted, 0);
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], 0);
}
