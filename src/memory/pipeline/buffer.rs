// src/memory/pipeline/buffer.rs
use anyhow::Result;
use chrono::Utc;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// A buffered message from the conversation.
#[derive(Debug, Clone)]
pub struct BufferedMessage {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub char_count: usize,
    pub created_at: String,
}

impl BufferedMessage {
    /// Format a slice of messages as conversation text.
    pub fn format_slice(messages: &[BufferedMessage]) -> String {
        messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Manages the memcell_buffer table — accumulates messages until boundary flush.
pub struct Buffer {
    conn: Arc<Mutex<Connection>>,
}

impl Buffer {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Append a message to the buffer.
    pub async fn append(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let id = Uuid::new_v4().to_string();
        let char_count = content.len() as i64;
        let created_at = Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO memcell_buffer (id, session_id, role, content, char_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![id, session_id, role, content, char_count, created_at],
        )?;
        Ok(())
    }

    /// Read all buffered messages for a session, ordered by time.
    pub async fn read(&self, session_id: &str) -> Result<Vec<BufferedMessage>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, role, content, char_count, created_at
             FROM memcell_buffer
             WHERE session_id = ?1
             ORDER BY created_at ASC"
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            Ok(BufferedMessage {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                char_count: row.get::<_, i64>(4)? as usize,
                created_at: row.get(5)?,
            })
        })?;
        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        Ok(messages)
    }

    /// Clear all buffered messages for a session.
    pub async fn clear(&self, session_id: &str) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM memcell_buffer WHERE session_id = ?1",
            rusqlite::params![session_id],
        )?;
        Ok(())
    }

    /// Count messages in buffer for a session.
    pub async fn message_count(&self, session_id: &str) -> Result<usize> {
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memcell_buffer WHERE session_id = ?1",
            rusqlite::params![session_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Estimate token count for a session (chars / 4).
    pub async fn estimated_tokens(&self, session_id: &str) -> Result<usize> {
        let conn = self.conn.lock().await;
        let total_chars: i64 = conn.query_row(
            "SELECT COALESCE(SUM(char_count), 0) FROM memcell_buffer WHERE session_id = ?1",
            rusqlite::params![session_id],
            |row| row.get(0),
        )?;
        Ok((total_chars as usize) / 4)
    }

    /// Delete stale buffer entries older than `ttl_hours`.
    /// Only deletes rows — does NOT run extraction first.
    pub async fn cleanup_stale(&self, ttl_hours: u64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::hours(ttl_hours as i64);
        let cutoff_str = cutoff.to_rfc3339();
        let conn = self.conn.lock().await;
        let deleted = conn.execute(
            "DELETE FROM memcell_buffer WHERE session_id IN (
                SELECT session_id FROM memcell_buffer
                GROUP BY session_id
                HAVING MAX(created_at) < ?1
            )",
            rusqlite::params![cutoff_str],
        )?;
        Ok(deleted)
    }
}
