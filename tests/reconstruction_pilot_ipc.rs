use ncm::rpc::{EngineClient, EngineConfig, ForwardRequest, InitModelRequest, TensorBatch};
use serde_json::json;
use std::path::Path;
#[test]
fn reconstruction_metrics_cross_the_jax_ipc_boundary() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut engine = EngineClient::start(root).unwrap();
    let config:EngineConfig=serde_json::from_value(json!({"rvq":{"lengths":[2,4,8],"codebook_size":8,"dim":4},"transformer":{"width":8,"heads":2,"layers":1,"mlp_ratio":2},"init_batch_size":1})).unwrap();
    let handle = engine
        .init_model(&InitModelRequest { config, seed: 3 })
        .unwrap();
    let batch = TensorBatch {
        tokens: vec![vec![1, 2, 3, 4, 5, 6, 7, 8]],
        mask: None,
        fallback_bytes: None,
        fallback_mask: None,
        prompt_ids: None,
        prompt_mask: None,
    };
    let result = engine
        .evaluate_reconstruction(&ForwardRequest {
            engine_id: handle.engine_id,
            batch,
        })
        .unwrap();
    assert!((0.0..=1.0).contains(&result.ber));
    assert!((0.0..=1.0).contains(&result.exact_match));
    assert!(!result.codebook_perplexity.is_empty());
    assert!(
        result
            .codebook_perplexity
            .iter()
            .all(|x| x.is_finite() && *x >= 1.0)
    );
}
