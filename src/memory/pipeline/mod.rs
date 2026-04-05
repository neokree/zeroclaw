// src/memory/pipeline/mod.rs
pub mod config;
pub mod buffer;
pub mod store;
pub mod boundary;
pub mod extractor;
pub mod formatter;
pub mod profile_types;
pub mod merger;

pub use config::PipelineConfig;

use anyhow::Result;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::memory::traits::Memory;
use crate::providers::traits::Provider;

use self::boundary::{BoundaryDetector, FlushDecision};
use self::buffer::Buffer;
use self::extractor::episode::EpisodeExtractor;
use self::extractor::event_log::EventLogExtractor;
use self::extractor::foresight::ForesightExtractor;
use self::extractor::profile;
use self::formatter::Formatter;
use self::merger::merge_profiles;
use self::profile_types::ProfileData;
use self::store::PipelineStore;

/// Memory Pipeline — boundary detection, extraction, and XML injection.
pub struct Pipeline {
    provider: Arc<dyn Provider>,
    default_model: String,
    buffer: Buffer,
    boundary: BoundaryDetector,
    episode_extractor: EpisodeExtractor,
    event_log_extractor: EventLogExtractor,
    foresight_extractor: ForesightExtractor,
    formatter: Formatter,
    store: Arc<PipelineStore>,
    config: PipelineConfig,
}

impl Pipeline {
    /// Create a new Pipeline.
    /// `default_model`: the main model from `[provider]` config, used as fallback
    /// when `extraction_model` is not set.
    pub fn new(
        provider: Arc<dyn Provider>,
        memory: Arc<dyn Memory>,
        store: PipelineStore,
        config: PipelineConfig,
        default_model: String,
    ) -> Self {
        let store = Arc::new(store);
        let conn = store.connection();
        Self {
            provider: provider.clone(),
            default_model: default_model.clone(),
            buffer: Buffer::new(conn),
            boundary: BoundaryDetector::new(provider.clone(), config.clone(), default_model.clone()),
            episode_extractor: EpisodeExtractor::new(provider.clone(), config.clone(), default_model.clone()),
            event_log_extractor: EventLogExtractor::new(provider.clone(), config.clone(), default_model.clone()),
            foresight_extractor: ForesightExtractor::new(provider, config.clone(), default_model),
            formatter: Formatter::new(store.clone(), memory, config.clone()),
            store,
            config,
        }
    }

    /// Append a message to the session buffer.
    pub async fn append_to_buffer(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
    ) -> Result<()> {
        self.buffer.append(session_id, role, content).await
    }

    /// Build XML memory context for injection. Called before LLM call.
    pub async fn build_context(
        &self,
        user_query: &str,
        min_relevance_score: f64,
        session_id: Option<&str>,
        user_id: Option<&str>,
    ) -> String {
        self.formatter
            .build_xml_context(user_query, min_relevance_score, session_id, user_id)
            .await
            .unwrap_or_default()
    }

    /// Post-turn processing: boundary check → extraction → save.
    /// Call via tokio::spawn (fire-and-forget).
    pub async fn process_turn(
        &self,
        session_id: &str,
        user_id: &str,
    ) {
        if let Err(e) = self.process_turn_inner(session_id, user_id).await {
            error!(
                target: "memory::pipeline",
                "Pipeline process_turn failed for session {session_id}: {e}"
            );
        }
    }

    async fn process_turn_inner(
        &self,
        session_id: &str,
        user_id: &str,
    ) -> Result<()> {
        let messages = self.buffer.read(session_id).await?;
        if messages.is_empty() {
            return Ok(());
        }

        // Boundary check
        let decision = match self.boundary.should_flush(&messages).await {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    target: "memory::pipeline",
                    "Boundary detection failed, skipping flush: {e}"
                );
                return Ok(());
            }
        };

        let reason = match decision {
            FlushDecision::Flush { reason } => reason,
            FlushDecision::Wait => return Ok(()),
        };

        info!(
            target: "memory::pipeline",
            "Flushing buffer for session {session_id}: {reason}"
        );

        // Run Episode + EventLog extraction in parallel
        let (episode_result, event_log_result) = tokio::join!(
            self.episode_extractor.extract(&messages),
            self.event_log_extractor.extract(&messages),
        );

        // Episode is required — if it fails, abort flush
        let episode_data = match episode_result {
            Ok(ep) => ep,
            Err(e) => {
                error!(
                    target: "memory::pipeline",
                    "Episode extraction failed, buffer preserved: {e}"
                );
                return Ok(());
            }
        };

        // Save episode
        let saved = self.store
            .save_episode(session_id, user_id, &episode_data)
            .await?;

        info!(
            target: "memory::pipeline",
            "Saved episode {}: {}",
            saved.id, saved.subject
        );

        // Foresight extraction — runs after episode is saved (needs saved.id + episode_data.summary)
        let buffer_text = buffer::BufferedMessage::format_slice(&messages);
        let foresight_entries = match self.foresight_extractor.extract(&buffer_text, &episode_data.summary).await {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(
                    target: "memory::pipeline",
                    "Foresight extraction failed (non-blocking): {e}"
                );
                vec![]
            }
        };

        // Save foresights (non-blocking)
        if !foresight_entries.is_empty() {
            use crate::memory::pipeline::store::ForesightRow;
            let foresight_rows: Vec<ForesightRow> = foresight_entries.iter().map(|f| ForesightRow {
                id: uuid::Uuid::new_v4().to_string(),
                episode_id: Some(saved.id.clone()),
                user_id: Some(user_id.to_string()),
                content: f.content.clone(),
                evidence: f.evidence.clone(),
                start_time: f.start_time.clone(),
                end_time: f.end_time.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
            }).collect();

            if let Err(e) = self.store.save_foresights(&foresight_rows).await {
                tracing::warn!(
                    target: "memory::pipeline",
                    "Failed to save foresights (non-blocking): {e}"
                );
            } else {
                tracing::info!(
                    target: "memory::pipeline",
                    "Saved {} foresights for episode {}",
                    foresight_rows.len(), saved.id
                );
            }
        }

        // Save event logs (non-blocking: failure does NOT block buffer clear)
        match event_log_result {
            Ok(event_logs) => {
                match self.store.save_event_logs(&saved.id, &event_logs.atomic_facts).await {
                    Ok(count) => {
                        info!(
                            target: "memory::pipeline",
                            "Saved {count} atomic facts for episode {}",
                            saved.id
                        );
                    }
                    Err(e) => {
                        warn!(
                            target: "memory::pipeline",
                            "EventLog INSERT failed (episode saved, buffer will clear): {e}"
                        );
                    }
                }
            }
            Err(e) => {
                warn!(
                    target: "memory::pipeline",
                    "EventLog extraction failed (episode saved, buffer will clear): {e}"
                );
            }
        }

        // --- Profile trigger (Phase 3) ---
        // Increment episode_count AFTER saving episode+event_logs+foresights.
        // If save succeeded but counter increment fails, profile trigger may be
        // delayed by one episode — acceptable for a best-effort feature.
        let episode_count = match self.store.increment_episode_count(user_id).await {
            Ok(count) => count,
            Err(e) => {
                tracing::warn!(
                    target: "memory::pipeline",
                    "Failed to increment episode count (non-blocking): {e}"
                );
                0
            }
        };

        if episode_count > 0
            && episode_count as usize % self.config.profile_update_interval == 0
        {
            let recent_episodes = self
                .store
                .get_recent_episodes(user_id, self.config.profile_update_interval)
                .await
                .unwrap_or_default();

            if !recent_episodes.is_empty() {
                let current_profile = match self.store.get_profile(user_id).await {
                    Ok(Some(row)) => serde_json::from_str::<ProfileData>(&row.profile_data)
                        .unwrap_or_default(),
                    _ => ProfileData::default(),
                };

                let extraction_model = self
                    .config
                    .extraction_model
                    .as_deref()
                    .unwrap_or(&self.default_model);

                match profile::extract(
                    self.provider.as_ref(),
                    extraction_model,
                    &current_profile,
                    &recent_episodes,
                    &self.config.prompt_language,
                )
                .await
                {
                    Ok(new_profile) => {
                        let merged = merge_profiles(&current_profile, &new_profile);
                        let merged_json = serde_json::to_string(&merged).unwrap_or_default();
                        if let Err(e) = self
                            .store
                            .upsert_profile(user_id, &merged_json, 0.0, &saved.id)
                            .await
                        {
                            tracing::warn!(
                                target: "memory::pipeline",
                                "Failed to save profile (non-blocking): {e}"
                            );
                        } else {
                            tracing::info!(
                                target: "memory::pipeline",
                                "Profile updated for user {user_id} at episode count {episode_count}"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "memory::pipeline",
                            "Profile extraction failed (non-blocking): {e}"
                        );
                    }
                }
            }
        }

        // Clear buffer — safe because episode was saved successfully above.
        self.buffer.clear(session_id).await?;

        Ok(())
    }

    /// Check if pipeline is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}
