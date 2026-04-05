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

    CREATE TABLE IF NOT EXISTS foresights (
        id            TEXT PRIMARY KEY,
        episode_id    TEXT REFERENCES episodes(id),
        user_id       TEXT,
        content       TEXT NOT NULL,
        evidence      TEXT NOT NULL,
        start_time    TEXT,
        end_time      TEXT,
        embedding     BLOB,
        created_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_foresights_episode ON foresights(episode_id);
    CREATE INDEX IF NOT EXISTS idx_foresights_end_time ON foresights(end_time);

    CREATE TABLE IF NOT EXISTS profiles (
        id              TEXT PRIMARY KEY,
        user_id         TEXT NOT NULL UNIQUE,
        profile_data    TEXT NOT NULL,
        confidence      REAL DEFAULT 0.0,
        version         INTEGER DEFAULT 1,
        episode_count   INTEGER DEFAULT 0,
        last_episode_id TEXT,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id);
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

    pub async fn save_foresights(&self, foresights: &[ForesightRow]) -> Result<()> {
        let conn = self.conn.lock().await;
        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO foresights (id, episode_id, user_id, content, evidence, start_time, end_time, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for f in foresights {
                stmt.execute(rusqlite::params![
                    f.id, f.episode_id, f.user_id, f.content, f.evidence,
                    f.start_time, f.end_time, f.created_at,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // TODO: replace created_at ordering with embedding similarity when available
    pub async fn get_active_foresights(
        &self,
        user_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ForesightRow>> {
        let conn = self.conn.lock().await;
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let rows: Vec<ForesightRow> = if let Some(uid) = user_id {
            let mut stmt = conn.prepare_cached(
                "SELECT id, episode_id, user_id, content, evidence, start_time, end_time, created_at
                 FROM foresights
                 WHERE (end_time IS NULL OR end_time >= ?1)
                   AND user_id = ?2
                 ORDER BY created_at DESC
                 LIMIT ?3",
            )?;
            stmt.query_map(rusqlite::params![today, uid, limit as i64], |row| {
                Ok(ForesightRow {
                    id: row.get(0)?,
                    episode_id: row.get(1)?,
                    user_id: row.get(2)?,
                    content: row.get(3)?,
                    evidence: row.get(4)?,
                    start_time: row.get(5)?,
                    end_time: row.get(6)?,
                    created_at: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut stmt = conn.prepare_cached(
                "SELECT id, episode_id, user_id, content, evidence, start_time, end_time, created_at
                 FROM foresights
                 WHERE (end_time IS NULL OR end_time >= ?1)
                 ORDER BY created_at DESC
                 LIMIT ?2",
            )?;
            stmt.query_map(rusqlite::params![today, limit as i64], |row| {
                Ok(ForesightRow {
                    id: row.get(0)?,
                    episode_id: row.get(1)?,
                    user_id: row.get(2)?,
                    content: row.get(3)?,
                    evidence: row.get(4)?,
                    start_time: row.get(5)?,
                    end_time: row.get(6)?,
                    created_at: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?
        };
        Ok(rows)
    }

    pub async fn get_profile(&self, user_id: &str) -> Result<Option<ProfileRow>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare_cached(
            "SELECT id, user_id, profile_data, confidence, version, episode_count,
                    last_episode_id, created_at, updated_at
             FROM profiles WHERE user_id = ?1",
        )?;
        let result = stmt.query_row(rusqlite::params![user_id], |row| {
            Ok(ProfileRow {
                id: row.get(0)?,
                user_id: row.get(1)?,
                profile_data: row.get(2)?,
                confidence: row.get(3)?,
                version: row.get(4)?,
                episode_count: row.get(5)?,
                last_episode_id: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        });
        match result {
            Ok(row) => Ok(Some(row)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn upsert_profile(
        &self,
        user_id: &str,
        profile_data_json: &str,
        confidence: f64,
        last_episode_id: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO profiles (id, user_id, profile_data, confidence, version, episode_count, last_episode_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 1, 1, ?5, ?6, ?6)
             ON CONFLICT(user_id) DO UPDATE SET
               profile_data = ?3,
               confidence = ?4,
               version = version + 1,
               last_episode_id = ?5,
               updated_at = ?6",
            rusqlite::params![Uuid::new_v4().to_string(), user_id, profile_data_json, confidence, last_episode_id, now],
        )?;
        Ok(())
    }

    /// Increment episode_count for a user. Returns the new count.
    /// Creates the profile row if it doesn't exist (with empty profile_data).
    pub async fn increment_episode_count(&self, user_id: &str) -> Result<i64> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO profiles (id, user_id, profile_data, confidence, version, episode_count, created_at, updated_at)
             VALUES (?1, ?2, '{}', 0.0, 1, 1, ?3, ?3)
             ON CONFLICT(user_id) DO UPDATE SET
               episode_count = episode_count + 1,
               updated_at = ?3",
            rusqlite::params![Uuid::new_v4().to_string(), user_id, now],
        )?;
        let count: i64 = conn.query_row(
            "SELECT episode_count FROM profiles WHERE user_id = ?1",
            rusqlite::params![user_id],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Retrieves the last N episodes for a given user_id, to pass to the profile extractor.
    pub async fn get_recent_episodes(&self, user_id: &str, limit: usize) -> Result<Vec<SavedEpisode>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare_cached(
            "SELECT id, subject, summary, episode, created_at
             FROM episodes
             WHERE user_id = ?1
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![user_id, limit as i64], |row| {
            Ok(SavedEpisode {
                id: row.get(0)?,
                subject: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                summary: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                episode: row.get(3)?,
                created_at: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
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

#[derive(Debug, Clone)]
pub struct ForesightRow {
    pub id: String,
    pub episode_id: Option<String>,
    pub user_id: Option<String>,
    pub content: String,
    pub evidence: String,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub created_at: String,
    // embedding: BLOB omitted — will be populated when embedding pipeline is added
}

#[derive(Debug, Clone)]
pub struct ProfileRow {
    pub id: String,
    pub user_id: String,
    pub profile_data: String, // JSON string
    pub confidence: f64,
    pub version: i64,
    pub episode_count: i64,
    pub last_episode_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}
