// src/memory/pipeline/store.rs
use anyhow::Result;
use chrono::Utc;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::memory::pipeline::extractor::{AtomicFact, EpisodeData};

const PIPELINE_SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS memcell_buffer (
        id          TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        char_count  INTEGER NOT NULL,
        created_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memcell_buffer_session
        ON memcell_buffer(session_id, created_at);

    CREATE TABLE IF NOT EXISTS episodes (
        id          TEXT PRIMARY KEY,
        session_id  TEXT,
        user_id     TEXT,
        subject     TEXT,
        summary     TEXT,
        episode     TEXT NOT NULL,
        embedding   BLOB,
        created_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS event_logs (
        id          TEXT PRIMARY KEY,
        episode_id  TEXT REFERENCES episodes(id),
        atomic_fact TEXT NOT NULL,
        time_ref    TEXT,
        embedding   BLOB,
        created_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_event_logs_episode
        ON event_logs(episode_id);
";

/// Manages pipeline-specific tables in brain.db.
/// Separate from the Memory trait — pipeline owns its own schema.
pub struct PipelineStore {
    conn: Arc<Mutex<Connection>>,
}

impl PipelineStore {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Create pipeline tables if they don't exist (async version).
    pub async fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute_batch(PIPELINE_SCHEMA)?;
        Ok(())
    }

    /// Synchronous schema init for use during SQLite backend construction.
    pub fn init_schema_sync(conn: &Connection) -> Result<()> {
        conn.execute_batch(PIPELINE_SCHEMA)?;
        Ok(())
    }

    /// Get a reference to the shared connection for sub-components.
    pub fn connection(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    /// Save an extracted episode and its event logs.
    /// Returns the saved episode with generated ID.
    pub async fn save_episode(
        &self,
        session_id: &str,
        user_id: &str,
        episode: &EpisodeData,
    ) -> Result<SavedEpisode> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO episodes (id, session_id, user_id, subject, summary, episode, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![id, session_id, user_id, episode.subject, episode.summary, episode.episode, now],
        )?;
        Ok(SavedEpisode {
            id,
            subject: episode.subject.clone(),
            episode: episode.episode.clone(),
            summary: episode.summary.clone(),
            created_at: now,
        })
    }

    /// Save extracted atomic facts linked to an episode.
    pub async fn save_event_logs(
        &self,
        episode_id: &str,
        facts: &[AtomicFact],
    ) -> Result<usize> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        let mut count = 0;
        for fact in facts {
            let id = Uuid::new_v4().to_string();
            conn.execute(
                "INSERT INTO event_logs (id, episode_id, atomic_fact, time_ref, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![id, episode_id, fact.fact, fact.time, now],
            )?;
            count += 1;
        }
        Ok(count)
    }

    /// Read the last N episodes, ordered by date descending.
    pub async fn recent_episodes(&self, limit: usize) -> Result<Vec<SavedEpisode>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, subject, summary, episode, created_at FROM episodes
             ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok(SavedEpisode {
                id: row.get(0)?,
                subject: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                summary: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                episode: row.get(3)?,
                created_at: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            })
        })?;
        let mut episodes = Vec::new();
        for row in rows {
            episodes.push(row?);
        }
        Ok(episodes)
    }
}

/// A saved episode with its generated ID.
#[derive(Debug, Clone)]
pub struct SavedEpisode {
    pub id: String,
    pub subject: String,
    pub episode: String,
    pub summary: String,
    pub created_at: String,
}
