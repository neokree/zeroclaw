// src/memory/pipeline/extractor/episode.rs
use anyhow::Result;
use std::sync::Arc;

use super::EpisodeData;
use crate::memory::pipeline::buffer::BufferedMessage;
use crate::memory::pipeline::config::PipelineConfig;
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};

pub struct EpisodeExtractor {
    provider: Arc<dyn Provider>,
    config: PipelineConfig,
    default_model: String,
}

impl EpisodeExtractor {
    pub fn new(provider: Arc<dyn Provider>, config: PipelineConfig, default_model: String) -> Self {
        Self { provider, config, default_model }
    }

    pub async fn extract(&self, messages: &[BufferedMessage]) -> Result<EpisodeData> {
        let prompt_template = self.get_prompt();
        let conversation = BufferedMessage::format_slice(messages);
        let prompt = prompt_template.replace("{conversation}", &conversation);

        let model = self.config.extraction_model.as_deref().unwrap_or(&self.default_model);

        let user_msg = ChatMessage::user(prompt);
        let request = ChatRequest {
            messages: &[user_msg],
            tools: None,
        };

        let response = self.provider.chat(request, model, 0.1).await?;
        let text = response.text.unwrap_or_default();

        serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("Episode extraction JSON parse error: {e}\nRaw: {text}"))
    }

    fn get_prompt(&self) -> &'static str {
        match self.config.prompt_language.as_str() {
            "it" => include_str!("../prompts/it/episode.txt"),
            _ => include_str!("../prompts/en/episode.txt"),
        }
    }
}
