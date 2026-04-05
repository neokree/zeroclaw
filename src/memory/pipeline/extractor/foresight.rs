// src/memory/pipeline/extractor/foresight.rs
use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;

use crate::memory::pipeline::config::PipelineConfig;
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};

#[derive(Debug, Clone)]
pub struct ForesightEntry {
    pub content: String,
    pub evidence: String,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

#[derive(Deserialize)]
struct ForesightResponse {
    foresights: Vec<ForesightItem>,
}

#[derive(Deserialize)]
struct ForesightItem {
    content: String,
    evidence: String,
    start_time: Option<String>,
    end_time: Option<String>,
}

pub struct ForesightExtractor {
    provider: Arc<dyn Provider>,
    config: PipelineConfig,
    default_model: String,
}

impl ForesightExtractor {
    pub fn new(provider: Arc<dyn Provider>, config: PipelineConfig, default_model: String) -> Self {
        Self { provider, config, default_model }
    }

    pub async fn extract(
        &self,
        buffer_content: &str,
        episode_summary: &str,
    ) -> Result<Vec<ForesightEntry>> {
        let template = match self.config.prompt_language.as_str() {
            "it" => include_str!("../prompts/it/foresight.txt"),
            _ => include_str!("../prompts/en/foresight.txt"),
        };

        let prompt = template
            .replace("{episode_summary}", episode_summary)
            .replace("{buffer_content}", buffer_content);

        let model = self.config.extraction_model.as_deref().unwrap_or(&self.default_model);

        let messages = [
            ChatMessage::system("You are a memory foresight engine. Extract behavioral predictions."),
            ChatMessage::user(&prompt),
        ];

        let request = ChatRequest {
            messages: &messages,
            tools: None,
        };

        let response = self.provider.chat(request, model, 0.1).await?;
        let text = response.text.unwrap_or_default();
        let text = text.trim();

        let parsed: ForesightResponse = serde_json::from_str(text)
            .map_err(|e| anyhow::anyhow!("Foresight JSON parse error: {e}\nRaw: {text}"))?;

        Ok(parsed
            .foresights
            .into_iter()
            .map(|f| ForesightEntry {
                content: f.content,
                evidence: f.evidence,
                start_time: f.start_time,
                end_time: f.end_time,
            })
            .collect())
    }
}
