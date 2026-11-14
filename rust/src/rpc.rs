//! JSON-RPC 2.0 messages used by the JAX worker's line-framed stdio protocol.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    env,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Request {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Response {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl Request {
    pub fn new(id: Value, method: impl Into<String>, params: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        }
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.jsonrpc != "2.0" {
            return Err("jsonrpc must be 2.0");
        }
        if !(self.id.is_string() || self.id.is_number()) {
            return Err("request id must be a string or number");
        }
        if self.method.is_empty() {
            return Err("method must be nonempty");
        }
        if !(self.params.is_object() || self.params.is_array() || self.params.is_null()) {
            return Err("params must be object, array, or null");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RvqConfig {
    pub lengths: Vec<usize>,
    pub codebook_size: usize,
    pub dim: usize,
    pub decay: f32,
    pub epsilon: f32,
    pub entropy_temperature: f32,
    pub nested_dropout: f32,
    pub coarse: Value,
}
impl Default for RvqConfig {
    fn default() -> Self {
        Self {
            lengths: vec![1, 2, 4],
            codebook_size: 8,
            dim: 4,
            decay: 0.99,
            epsilon: 1e-5,
            entropy_temperature: 1.0,
            nested_dropout: 0.,
            coarse: json!({}),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TransformerConfig {
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub mlp_ratio: usize,
    pub dropout: f32,
}
impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            width: 256,
            heads: 8,
            layers: 4,
            mlp_ratio: 4,
            dropout: 0.0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelConfig {
    pub ar_coarse_levels: usize,
    pub prompt_len: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LossConfig {
    pub beta: f32,
    pub entropy: f32,
    pub concept: f32,
    pub prior: f32,
    pub entropy_weights: Vec<f32>,
}
impl Default for LossConfig {
    fn default() -> Self {
        Self {
            beta: 0.25,
            entropy: 0.01,
            concept: 0.0,
            prior: 1.0,
            entropy_weights: vec![],
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    pub rvq: RvqConfig,
    #[serde(default)]
    pub transformer: TransformerConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub losses: LossConfig,
    #[serde(default = "default_vocab")]
    pub token_vocab_size: usize,
    #[serde(default = "default_lr")]
    pub learning_rate: f32,
    #[serde(default = "default_batch")]
    pub init_batch_size: usize,
    #[serde(default)]
    pub device: DeviceConfig,
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DeviceConfig {
    pub platform: Option<String>,
    pub index: usize,
}
fn default_vocab() -> usize {
    256
}
fn default_lr() -> f32 {
    3e-4
}
fn default_batch() -> usize {
    1
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitModelRequest {
    pub config: EngineConfig,
    pub seed: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorBatch {
    pub tokens: Vec<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<Vec<Vec<bool>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_bytes: Option<Vec<Vec<i32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_mask: Option<Vec<Vec<bool>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_ids: Option<Vec<Vec<i32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_mask: Option<Vec<Vec<bool>>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelHandle {
    pub engine_id: String,
    pub step: u64,
    pub backend: String,
    pub devices: Vec<String>,
    pub sharding: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForwardRequest {
    pub engine_id: String,
    pub batch: TensorBatch,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForwardResponse {
    pub loss: f64,
    pub metrics: Value,
    pub codes: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainStepRequest {
    pub engine_id: String,
    pub batch: TensorBatch,
    pub seed: u32,
    #[serde(default)]
    pub phase: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainStepResponse {
    pub step: u64,
    pub loss: f64,
    pub grad_norm: f64,
    pub step_time_ms: f64,
    pub metrics: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointRequest {
    pub engine_id: String,
    pub path: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointResponse {
    pub path: String,
    pub step: u64,
}

pub struct EngineClient {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<std::process::ChildStdout>,
    next_id: u64,
}
impl EngineClient {
    pub fn start(root: &Path) -> Result<Self, String> {
        let python = env::var_os("NCM_PYTHON").unwrap_or_else(|| "python3".into());
        let mut paths = vec![root.join("src")];
        if let Some(old) = env::var_os("PYTHONPATH") {
            paths.extend(env::split_paths(&old));
        }
        let mut child = Command::new(python)
            .args(["-u", "-m", "ncm.engine_rpc"])
            .current_dir(root)
            .env(
                "PYTHONPATH",
                env::join_paths(paths).map_err(|e| e.to_string())?,
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("could not start JAX engine: {e}"))?;
        Ok(Self {
            input: BufWriter::new(child.stdin.take().ok_or("engine stdin unavailable")?),
            output: BufReader::new(child.stdout.take().ok_or("engine stdout unavailable")?),
            child,
            next_id: 1,
        })
    }
    pub fn call<P: Serialize, R: for<'de> Deserialize<'de>>(
        &mut self,
        method: &str,
        params: &P,
    ) -> Result<R, String> {
        if let Some(status) = self
            .child
            .try_wait()
            .map_err(|e| format!("could not inspect JAX engine status: {e}"))?
        {
            return Err(format!("JAX engine exited unexpectedly: {status}"));
        }
        let id = self.next_id;
        self.next_id += 1;
        let req = Request::new(json!(id), method, json!(params));
        req.validate().map_err(str::to_owned)?;
        serde_json::to_writer(&mut self.input, &req).map_err(|e| e.to_string())?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("engine request write failed: {e}"))?;
        let mut line = String::new();
        let n = self
            .output
            .read_line(&mut line)
            .map_err(|e| format!("engine response read failed: {e}"))?;
        if n == 0 {
            return Err(format!(
                "engine exited before replying (status {:?})",
                self.child.try_wait().ok().flatten()
            ));
        }
        let response: Response =
            serde_json::from_str(&line).map_err(|e| format!("invalid JSON-RPC response: {e}"))?;
        if response.jsonrpc != "2.0" || response.id != json!(id) {
            return Err("JSON-RPC version or request ID mismatch".into());
        }
        if let Some(error) = response.error {
            return Err(format!("JAX engine RPC {}: {}", error.code, error.message));
        }
        serde_json::from_value(response.result.ok_or("JSON-RPC response has no result")?)
            .map_err(|e| format!("invalid {method} result: {e}"))
    }
    pub fn init_model(&mut self, request: &InitModelRequest) -> Result<ModelHandle, String> {
        self.call("init_model", request)
    }
    pub fn describe(&mut self) -> Result<Value, String> {
        self.call("engine.describe", &json!({}))
    }
    pub fn forward(&mut self, request: &ForwardRequest) -> Result<ForwardResponse, String> {
        self.call("forward", request)
    }
    pub fn train_step(&mut self, request: &TrainStepRequest) -> Result<TrainStepResponse, String> {
        self.call("train_step", request)
    }
    pub fn save_checkpoint(
        &mut self,
        request: &CheckpointRequest,
    ) -> Result<CheckpointResponse, String> {
        self.call("save_checkpoint", request)
    }
    pub fn load_checkpoint(
        &mut self,
        request: &CheckpointRequest,
    ) -> Result<CheckpointResponse, String> {
        self.call("load_checkpoint", request)
    }
}
impl Drop for EngineClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
