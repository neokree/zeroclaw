// src/memory/pipeline/boundary.rs
use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;

use crate::memory::pipeline::buffer::BufferedMessage;
use crate::memory::pipeline::config::PipelineConfig;
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};

#[derive(Debug, Deserialize)]
pub struct BoundaryResult {
    pub reasoning: String,
    pub should_end: bool,
    pub confidence: f64,
    pub topic_summary: String,
}

pub enum FlushDecision {
    /// Flush the buffer — boundary detected or hard limit hit.
    Flush { reason: String },
    /// Keep accumulating — no boundary yet.
    Wait,
}

pub struct BoundaryDetector {
    provider: Arc<dyn Provider>,
    config: PipelineConfig,
    default_model: String,
}

impl BoundaryDetector {
    pub fn new(provider: Arc<dyn Provider>, config: PipelineConfig, default_model: String) -> Self {
        Self { provider, config, default_model }
    }

    /// Decide whether to flush the buffer.
    pub async fn should_flush(
        &self,
        messages: &[BufferedMessage],
    ) -> Result<FlushDecision> {
        if messages.is_empty() {
            return Ok(FlushDecision::Wait);
        }

        // Layer 1: Hard limits
        let total_chars: usize = messages.iter().map(|m| m.char_count).sum();
        let estimated_tokens = total_chars / 4;
        if estimated_tokens >= self.config.hard_token_limit {
            return Ok(FlushDecision::Flush {
                reason: format!("hard token limit ({estimated_tokens} >= {})",
                    self.config.hard_token_limit),
            });
        }
        if messages.len() >= self.config.hard_message_limit {
            return Ok(FlushDecision::Flush {
                reason: format!("hard message limit ({} >= {})",
                    messages.len(), self.config.hard_message_limit),
            });
        }

        // Need at least 2 messages for boundary detection
        if messages.len() < 2 {
            return Ok(FlushDecision::Wait);
        }

        // Layer 2: LLM semantic boundary detection
        let split_point = messages.len() - 1;
        let (history, new) = messages.split_at(split_point);

        let prompt_template = self.get_prompt();
        let history_text = BufferedMessage::format_slice(history);
        let new_text = BufferedMessage::format_slice(new);
        let prompt = prompt_template
            .replace("{history}", &history_text)
            .replace("{new_messages}", &new_text);

        let model = self.config.extraction_model.as_deref()
            .unwrap_or(&self.default_model);

        let user_msg = ChatMessage::user(prompt);
        let request = ChatRequest {
            messages: &[user_msg],
            tools: None,
        };

        let response = self.provider.chat(request, model, 0.1).await?;
        let text = response.text.unwrap_or_default();

        let result: BoundaryResult = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("Boundary detection JSON parse error: {e}\nRaw: {text}"))?;

        if result.should_end && result.confidence >= self.config.boundary_threshold {
            Ok(FlushDecision::Flush {
                reason: format!("boundary detected (confidence={:.2}): {}",
                    result.confidence, result.topic_summary),
            })
        } else {
            Ok(FlushDecision::Wait)
        }
    }

    fn get_prompt(&self) -> &'static str {
        match self.config.prompt_language.as_str() {
            "it" => include_str!("prompts/it/boundary.txt"),
            _ => include_str!("prompts/en/boundary.txt"),
        }
    }
}
