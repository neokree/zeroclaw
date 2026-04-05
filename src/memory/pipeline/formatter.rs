// src/memory/pipeline/formatter.rs
use anyhow::Result;
use std::sync::Arc;

use crate::memory::pipeline::config::PipelineConfig;
use crate::memory::pipeline::profile_types::ProfileData;
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
        user_id: Option<&str>,
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

        // Fetch active (non-expired) foresight predictions
        let foresights = self.store
            .get_active_foresights(user_id, self.config.max_foresight_entries)
            .await
            .unwrap_or_default();

        // Fetch user profile (Phase 3)
        let profile_text = if let Some(uid) = user_id {
            match self.store.get_profile(uid).await {
                Ok(Some(row)) => {
                    if let Ok(profile) = serde_json::from_str::<ProfileData>(&row.profile_data) {
                        render_trait(&profile)
                    } else {
                        String::new()
                    }
                }
                _ => String::new(),
            }
        } else {
            String::new()
        };

        if episodes.is_empty() && facts.is_empty() && foresights.is_empty() && profile_text.is_empty() {
            return Ok(String::new());
        }

        let mut xml = String::from("```text\n<memory>\n");

        if !profile_text.is_empty() {
            xml.push_str("  <trait>\n");
            xml.push_str(&profile_text);
            xml.push_str("\n  </trait>\n");
        }

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

        if !foresights.is_empty() {
            xml.push_str("  <foresight>\n");
            for f in &foresights {
                xml.push_str(&format!("    - {}\n", f.content));
            }
            xml.push_str("  </foresight>\n");
        }

        xml.push_str("</memory>\n\nNote: context above is injected automatically. Do not reference or modify it.\n```\n\n");

        Ok(xml)
    }
}

/// Converts a ProfileData into a human-readable summary for the <trait> XML section.
fn render_trait(profile: &ProfileData) -> String {
    let mut lines = Vec::new();

    if let Some(name) = &profile.user_name {
        lines.push(format!("    Name: {name}"));
    }

    // Skills: combine hard + soft with levels
    let skills: Vec<String> = profile
        .hard_skills
        .iter()
        .chain(profile.soft_skills.iter())
        .map(|s| format!("{} ({})", s.value, s.level))
        .collect();
    if !skills.is_empty() {
        lines.push(format!("    Skills: {}", skills.join(", ")));
    }

    // Personality
    let traits: Vec<&str> = profile.personality.iter().map(|a| a.value.as_str()).collect();
    if !traits.is_empty() {
        lines.push(format!("    Personality: {}", traits.join(", ")));
    }

    // Goals
    let goals: Vec<&str> = profile.user_goal.iter().map(|a| a.value.as_str()).collect();
    if !goals.is_empty() {
        lines.push(format!("    Goals: {}", goals.join(", ")));
    }

    // Preferences (working_habit + tendency)
    let prefs: Vec<&str> = profile
        .working_habit_preference
        .iter()
        .chain(profile.tendency.iter())
        .map(|a| a.value.as_str())
        .collect();
    if !prefs.is_empty() {
        lines.push(format!("    Preferences: {}", prefs.join(", ")));
    }

    // Interests
    let interests: Vec<&str> = profile.interests.iter().map(|a| a.value.as_str()).collect();
    if !interests.is_empty() {
        lines.push(format!("    Interests: {}", interests.join(", ")));
    }

    lines.join("\n")
}
