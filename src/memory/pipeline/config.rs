// src/memory/pipeline/config.rs
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PipelineConfig {
    pub enabled: bool,
    pub extraction_model: Option<String>,
    pub prompt_language: String,
    pub boundary_threshold: f64,
    pub hard_token_limit: usize,
    pub hard_message_limit: usize,
    pub buffer_ttl_hours: u64,
    pub max_episodic_entries: usize,
    pub max_facts_entries: usize,
    pub max_foresight_entries: usize,
    pub profile_update_interval: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            extraction_model: None,
            prompt_language: "en".into(),
            boundary_threshold: 0.6,
            hard_token_limit: 4096,
            hard_message_limit: 30,
            buffer_ttl_hours: 24,
            max_episodic_entries: 5,
            max_facts_entries: 5,
            max_foresight_entries: 3,
            profile_update_interval: 5,
        }
    }
}
