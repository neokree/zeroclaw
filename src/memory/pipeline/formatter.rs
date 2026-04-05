// src/memory/pipeline/formatter.rs
use anyhow::Result;
use std::sync::Arc;

use crate::memory::pipeline::config::PipelineConfig;
use crate::memory::pipeline::store::PipelineStore;
use crate::memory::traits::Memory;

pub struct Formatter {
    store: Arc<PipelineStore>,
    memory: Arc<dyn Memory>,
    config: PipelineConfig,
}

impl Formatter {
    pub fn new(
        store: Arc<PipelineStore>,
        memory: Arc<dyn Memory>,
        config: PipelineConfig,
    ) -> Self {
        Self { store, memory, config }
    }

    /// Build the XML memory context block.
    /// Returns empty string if no memories are available.
    pub async fn build_xml_context(
        &self,
        user_query: &str,
        min_relevance_score: f64,
        session_id: Option<&str>,
    ) -> Result<String> {
        // Fetch episodic memories (recency-based)
        let episodes = self.store
            .recent_episodes(self.config.max_episodic_entries)
            .await
            .unwrap_or_default();

        // Fetch facts via existing Memory::recall (relevance-based)
        let all_facts = self.memory
            .recall(user_query, self.config.max_facts_entries * 2, session_id, None, None)
            .await
            .unwrap_or_default();
        let facts: Vec<_> = all_facts
            .into_iter()
            .filter(|e| {
                e.score.unwrap_or(0.0) >= min_relevance_score
                    && !crate::memory::is_assistant_autosave_key(&e.key)
            })
            .take(self.config.max_facts_entries)
            .collect();

        if episodes.is_empty() && facts.is_empty() {
            return Ok(String::new());
        }

        let mut xml = String::from("```text\n<memory>\n");

        if !episodes.is_empty() {
            xml.push_str("  <episodic>\n");
            for (i, ep) in episodes.iter().rev().enumerate() {
                let display = if !ep.summary.is_empty() {
                    &ep.summary
                } else {
                    &ep.episode
                };
                let truncated = if display.chars().count() > 200 {
                    let s: String = display.chars().take(200).collect();
                    format!("{s}...")
                } else {
                    display.to_string()
                };
                let date = ep.created_at.get(..10).unwrap_or(&ep.created_at);
                xml.push_str(&format!("    [{}] ({}) {}\n", i + 1, date, truncated));
            }
            xml.push_str("  </episodic>\n");
        }

        if !facts.is_empty() {
            xml.push_str("  <facts>\n");
            for entry in &facts {
                xml.push_str(&format!("    - {}\n", entry.content));
            }
            xml.push_str("  </facts>\n");
        }

        xml.push_str("</memory>\n\nNote: context above is injected automatically. Do not reference or modify it.\n```\n\n");

        Ok(xml)
    }
}
