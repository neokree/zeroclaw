# Memory Pipeline Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add boundary detection, episode + event-log extraction, and XML memory injection to ZeroClaw.

**Architecture:** New `src/memory/pipeline/` module that hooks into `channels/mod.rs` (post-response fire-and-forget) and `agent/loop_.rs` (XML injection replacing `[Memory context]`). Uses existing `Provider::chat()` for LLM calls. Embedding generation deferred to Phase 2 (YAGNI — schema supports it, code doesn't populate it yet). All pipeline state lives in new SQLite tables alongside existing `memories` table. Pipeline opens its own SQLite connection to the same `brain.db` file.

**Tech Stack:** Rust, SQLite (rusqlite), tokio, serde_json, chrono

**Spec:** `docs/superpowers/specs/2026-04-05-zeroclaw-memory-pipeline-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/memory/pipeline/mod.rs` | `Pipeline` struct, `process_turn()` orchestration |
| `src/memory/pipeline/config.rs` | `PipelineConfig` deserialization from `[memory.pipeline]` |
| `src/memory/pipeline/buffer.rs` | `Buffer`: append/read/clear/count/estimate_tokens on `memcell_buffer` table |
| `src/memory/pipeline/store.rs` | `PipelineStore`: manages `memcell_buffer`, `episodes`, `event_logs` tables + schema init |
| `src/memory/pipeline/boundary.rs` | `BoundaryDetector`: hard limits + LLM semantic detection |
| `src/memory/pipeline/extractor/mod.rs` | `ExtractionResult` struct, orchestration of parallel extractors |
| `src/memory/pipeline/extractor/episode.rs` | `EpisodeExtractor`: LLM call → `{subject, episode, summary}` |
| `src/memory/pipeline/extractor/event_log.rs` | `EventLogExtractor`: LLM call → `{atomic_facts: [...]}` |
| `src/memory/pipeline/formatter.rs` | `Formatter`: builds XML `<memory>` block from episodes + Memory::recall |
| `src/memory/pipeline/prompts/en/boundary.txt` | English boundary detection prompt |
| `src/memory/pipeline/prompts/en/episode.txt` | English episode extraction prompt |
| `src/memory/pipeline/prompts/en/event_log.txt` | English event-log extraction prompt |
| `src/memory/pipeline/prompts/it/boundary.txt` | Italian boundary detection prompt |
| `src/memory/pipeline/prompts/it/episode.txt` | Italian episode extraction prompt |
| `src/memory/pipeline/prompts/it/event_log.txt` | Italian event-log extraction prompt |

### Modified Files

| File | Change |
|------|--------|
| `src/memory/mod.rs` | Add `pub mod pipeline;` |
| `src/memory/sqlite.rs:159-220` | Add pipeline table creation in `init_schema()` |
| `src/config/schema.rs:4842-4964` | Add `pipeline: Option<PipelineConfig>` to `MemoryConfig` |
| `src/channels/mod.rs:2457-2530,2622-2670,3158-3191` | Construct `Pipeline`, call `append_to_buffer` + `process_turn` |
| `src/agent/loop_.rs:291-338,3899,4177,4726` | Replace `[Memory context]` with `Formatter::build_xml_context()` |

---

## Chunk 1: Foundation — Config + Store + Buffer

### Task 1: PipelineConfig

**Files:**
- Create: `src/memory/pipeline/config.rs`

- [ ] **Step 1: Create config struct**

```rust
// src/memory/pipeline/config.rs
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
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
        }
    }
}
```

- [ ] **Step 2: Add pipeline field to MemoryConfig**

In `src/config/schema.rs`, add to the `MemoryConfig` struct (around line 4964, before the closing brace):

```rust
    /// Memory pipeline configuration (boundary detection, extraction, XML injection)
    #[serde(default)]
    pub pipeline: crate::memory::pipeline::PipelineConfig,
```

- [ ] **Step 3: Add `pub mod pipeline;` to memory module**

In `src/memory/mod.rs`, add:

```rust
pub mod pipeline;
```

- [ ] **Step 4: Create pipeline module entry point**

```rust
// src/memory/pipeline/mod.rs
pub mod config;
pub mod buffer;
pub mod store;

pub use config::PipelineConfig;
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check 2>&1 | head -20`
Expected: no errors (warnings OK)

- [ ] **Step 6: Commit**

```bash
git add src/memory/pipeline/ src/config/schema.rs src/memory/mod.rs
git commit -m "feat(memory): add PipelineConfig and pipeline module skeleton"
```

---

### Task 2: PipelineStore — Schema + Table Creation

**Files:**
- Create: `src/memory/pipeline/store.rs`
- Modify: `src/memory/sqlite.rs:161-220`

- [ ] **Step 1: Create PipelineStore with schema init**

```rust
// src/memory/pipeline/store.rs
use anyhow::Result;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Manages pipeline-specific tables in brain.db.
/// Separate from the Memory trait — pipeline owns its own schema.
pub struct PipelineStore {
    conn: Arc<Mutex<Connection>>,
}

impl PipelineStore {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Create pipeline tables if they don't exist.
    pub async fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memcell_buffer (
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
                ON event_logs(episode_id);"
        )?;
        Ok(())
    }

    /// Get a reference to the shared connection for sub-components.
    pub fn connection(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }
}
```

- [ ] **Step 2: Wire pipeline schema init into sqlite.rs**

In `src/memory/sqlite.rs`, find the `init_schema` function (line ~159). After the existing schema creation, add at the end of the function body (before `Ok(())`):

```rust
    // Pipeline tables (created unconditionally — lightweight, no-op if exist)
    crate::memory::pipeline::store::PipelineStore::init_schema_sync(conn)?;
```

Then add a sync version to `store.rs`:

```rust
impl PipelineStore {
    // ... existing methods ...

    /// Synchronous schema init for use during SQLite backend construction.
    pub fn init_schema_sync(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memcell_buffer (
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
                ON event_logs(episode_id);"
        )?;
        Ok(())
    }
}
```

**Note:** Extract the SQL into a `const PIPELINE_SCHEMA: &str` at the top of `store.rs`, shared by both `init_schema` (async) and `init_schema_sync`.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check 2>&1 | head -20`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add src/memory/pipeline/store.rs src/memory/sqlite.rs
git commit -m "feat(memory): add PipelineStore with schema init"
```

---

### Task 3: Buffer — Append, Read, Clear, Stats

**Files:**
- Create: `src/memory/pipeline/buffer.rs`

- [ ] **Step 1: Create Buffer struct with all operations**

```rust
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
    ///
    /// **Note**: This only deletes the buffer rows without running extraction.
    /// The spec calls for a force-flush + extract on abandoned sessions, but
    /// implementing that requires a reference to the full pipeline — which
    /// `Buffer` does not have. The nightly cleanup should be wired at the
    /// `Pipeline` level (not yet implemented): call `process_turn_inner` on
    /// each stale session before calling `cleanup_stale`. For now, stale
    /// buffers are silently discarded, which is acceptable for single-user
    /// deployments where sessions rarely go stale.
    pub async fn cleanup_stale(&self, ttl_hours: u64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::hours(ttl_hours as i64);
        let cutoff_str = cutoff.to_rfc3339();
        let conn = self.conn.lock().await;
        // Find sessions where the newest message is older than cutoff
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
```

- [ ] **Step 2: Export buffer from pipeline module**

In `src/memory/pipeline/mod.rs`, it should already have `pub mod buffer;`. Verify.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check 2>&1 | head -20`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add src/memory/pipeline/buffer.rs src/memory/pipeline/mod.rs
git commit -m "feat(memory): add Buffer for memcell_buffer operations"
```

---

## Chunk 2: Boundary Detection

### Task 4: Prompt Files

**Files:**
- Create: `src/memory/pipeline/prompts/en/boundary.txt`
- Create: `src/memory/pipeline/prompts/it/boundary.txt`
- Create: `src/memory/pipeline/prompts/en/episode.txt`
- Create: `src/memory/pipeline/prompts/it/episode.txt`
- Create: `src/memory/pipeline/prompts/en/event_log.txt`
- Create: `src/memory/pipeline/prompts/it/event_log.txt`

- [ ] **Step 1: Create English boundary prompt**

```text
Analyze this conversation and determine if the NEW MESSAGES represent
a topic boundary (end of a coherent conversation segment).

CONVERSATION HISTORY:
{history}

NEW MESSAGES:
{new_messages}

Decision criteria (priority order):
1. Substantive Topic Change: completely different topic with meaningful content?
2. Intent/Purpose Transition: core question resolved, new substantial topic beginning?
3. Meaningful Content: ignore greetings/small talk, focus on memorable content
4. Temporal Signals: significant time gap? explicit concluding statements?
5. Content Independence: how related is new content to previous discussion?

Special rules:
- Greeting + Topic = ONE episode (do not split)
- "By the way" / "Speaking of" = usually same episode
- Farewells/closures = end of current episode, do not start new one

Respond ONLY with valid JSON:
{"reasoning": "one sentence", "should_end": true, "confidence": 0.85, "topic_summary": "summary if should_end, else empty string"}
```

- [ ] **Step 2: Create Italian boundary prompt**

Same structure, translated to Italian.

- [ ] **Step 3: Create English episode prompt**

```text
Convert this conversation into a concise third-person narrative.

CONVERSATION:
{conversation}

Requirements:
- Chronological factual record with all important info
- Preserve: person names, dates, locations, emotions, decisions, outcomes
- Time references as: "relative time (absolute date)"
- Record frequency information (patterns, repetitions)
- Remove redundancies while keeping all facts
- Do NOT include greetings/small talk

Respond ONLY with valid JSON:
{"subject": "concise title (10-20 words)", "episode": "third-person narrative", "summary": "2-3 sentence abstract"}
```

- [ ] **Step 4: Create Italian episode prompt**

Same structure, translated to Italian.

- [ ] **Step 5: Create English event_log prompt**

```text
Extract atomic facts from this conversation.

CONVERSATION:
{conversation}

Rules:
- Each fact = exactly one coherent unit (action, emotion, plan, decision)
- Single complete sentence, third-person form
- Explicit attribution: always state WHO said/did what
- Resolve pronouns to specific names
- Preserve timestamps; resolve relative times with absolute dates
- Filter out: greetings, "okay", low-value chatter

Respond ONLY with valid JSON:
{"atomic_facts": [{"fact": "sentence", "time": "time reference or null"}]}
```

- [ ] **Step 6: Create Italian event_log prompt**

Same structure, translated to Italian.

- [ ] **Step 7: Commit**

```bash
git add src/memory/pipeline/prompts/
git commit -m "feat(memory): add localized pipeline prompts (en, it)"
```

---

### Task 5: BoundaryDetector

**Files:**
- Create: `src/memory/pipeline/boundary.rs`

- [ ] **Step 1: Create BoundaryDetector**

```rust
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
    default_model: String, // main model from [provider] config, used if extraction_model is None
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
        // Smart mask: always put the last message in `new` so the LLM
        // evaluates it as the new content. With >5 messages this prevents
        // the most recent exchange from biasing the history context.
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
        let request = ChatRequest {
            messages: &[ChatMessage::user(&prompt)],
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
```

- [ ] **Step 2: Add helper to BufferedMessage**

In `src/memory/pipeline/buffer.rs`, add to `BufferedMessage`:

```rust
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
```

**Note**: `Buffer` does not have a `format_for_prompt` method — use `BufferedMessage::format_slice` directly.

- [ ] **Step 3: Export boundary from pipeline module**

In `src/memory/pipeline/mod.rs`, add:

```rust
pub mod boundary;
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check 2>&1 | head -20`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add src/memory/pipeline/boundary.rs src/memory/pipeline/buffer.rs src/memory/pipeline/mod.rs
git commit -m "feat(memory): add BoundaryDetector with hard limits + LLM detection"
```

---

## Chunk 3: Extractors — Episode + EventLog

### Task 6: Episode + EventLog Extractors

**Files:**
- Create: `src/memory/pipeline/extractor/mod.rs`
- Create: `src/memory/pipeline/extractor/episode.rs`
- Create: `src/memory/pipeline/extractor/event_log.rs`

- [ ] **Step 1: Create extractor types**

```rust
// src/memory/pipeline/extractor/mod.rs
pub mod episode;
pub mod event_log;

use serde::Deserialize;

/// Result of episode extraction.
#[derive(Debug, Clone, Deserialize)]
pub struct EpisodeData {
    pub subject: String,
    pub episode: String,
    pub summary: String,
}

/// A single atomic fact.
#[derive(Debug, Clone, Deserialize)]
pub struct AtomicFact {
    pub fact: String,
    pub time: Option<String>,
}

/// Result of event-log extraction.
#[derive(Debug, Clone, Deserialize)]
pub struct EventLogData {
    pub atomic_facts: Vec<AtomicFact>,
}

/// Combined extraction results from a flush.
#[derive(Debug)]
pub struct ExtractionResult {
    pub episode: EpisodeData,
    pub event_logs: EventLogData,
}
```

- [ ] **Step 2: Create EpisodeExtractor**

```rust
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
        let request = ChatRequest {
            messages: &[ChatMessage::user(&prompt)],
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
```

- [ ] **Step 3: Create EventLogExtractor**

```rust
// src/memory/pipeline/extractor/event_log.rs
use anyhow::Result;
use std::sync::Arc;

use super::EventLogData;
use crate::memory::pipeline::buffer::BufferedMessage;
use crate::memory::pipeline::config::PipelineConfig;
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};

pub struct EventLogExtractor {
    provider: Arc<dyn Provider>,
    config: PipelineConfig,
    default_model: String,
}

impl EventLogExtractor {
    pub fn new(provider: Arc<dyn Provider>, config: PipelineConfig, default_model: String) -> Self {
        Self { provider, config, default_model }
    }

    pub async fn extract(&self, messages: &[BufferedMessage]) -> Result<EventLogData> {
        let prompt_template = self.get_prompt();
        let conversation = BufferedMessage::format_slice(messages);
        let prompt = prompt_template.replace("{conversation}", &conversation);

        let model = self.config.extraction_model.as_deref().unwrap_or(&self.default_model);
        let request = ChatRequest {
            messages: &[ChatMessage::user(&prompt)],
            tools: None,
        };

        let response = self.provider.chat(request, model, 0.1).await?;
        let text = response.text.unwrap_or_default();

        serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("EventLog extraction JSON parse error: {e}\nRaw: {text}"))
    }

    fn get_prompt(&self) -> &'static str {
        match self.config.prompt_language.as_str() {
            "it" => include_str!("../prompts/it/event_log.txt"),
            _ => include_str!("../prompts/en/event_log.txt"),
        }
    }
}
```

- [ ] **Step 4: Export extractor from pipeline module**

In `src/memory/pipeline/mod.rs`, add:

```rust
pub mod extractor;
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check 2>&1 | head -20`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/memory/pipeline/extractor/
git commit -m "feat(memory): add Episode and EventLog extractors"
```

---

### Task 7: PipelineStore — Save Episode + EventLogs

**Files:**
- Modify: `src/memory/pipeline/store.rs`

- [ ] **Step 1: Add save methods to PipelineStore**

Append to `src/memory/pipeline/store.rs`:

```rust
use crate::memory::pipeline::extractor::{EpisodeData, AtomicFact};
use uuid::Uuid;
use chrono::Utc;

/// A saved episode with its generated ID.
#[derive(Debug, Clone)]
pub struct SavedEpisode {
    pub id: String,
    pub subject: String,
    pub episode: String,
    pub summary: String,
    pub created_at: String,
}

impl PipelineStore {
    /// Save an extracted episode and its event logs.
    /// Returns the episode ID for linking.
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
            created_at: now.clone(),
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
             ORDER BY created_at DESC LIMIT ?1"
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
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check 2>&1 | head -20`

- [ ] **Step 3: Commit**

```bash
git add src/memory/pipeline/store.rs
git commit -m "feat(memory): add PipelineStore save/read methods for episodes + event_logs"
```

---

## Chunk 4: Formatter + Pipeline Orchestration + Integration

### Task 8: Formatter — XML Injection Builder

**Files:**
- Create: `src/memory/pipeline/formatter.rs`

- [ ] **Step 1: Create Formatter**

```rust
// src/memory/pipeline/formatter.rs
use anyhow::Result;
use std::sync::Arc;

use crate::memory::pipeline::config::PipelineConfig;
use crate::memory::pipeline::store::{PipelineStore, SavedEpisode};
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

    /// Build the XML memory context block to prepend to the user message.
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
        // Over-fetch then filter by min_relevance_score
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

        // <episodic> section
        if !episodes.is_empty() {
            xml.push_str("  <episodic>\n");
            for (i, ep) in episodes.iter().rev().enumerate() {
                // Extract date from created_at or use subject
                let display = if !ep.summary.is_empty() {
                    &ep.summary
                } else {
                    &ep.episode
                };
                // Truncate to ~200 chars for injection
                let truncated = if display.len() > 200 {
                    format!("{}...", &display[..200])
                } else {
                    display.to_string()
                };
                // Extract date portion from RFC3339 created_at (e.g. "2026-04-05")
                let date = ep.created_at.get(..10).unwrap_or(&ep.created_at);
                xml.push_str(&format!("    [{}] ({}) {}\n", i + 1, date, truncated));
            }
            xml.push_str("  </episodic>\n");
        }

        // <facts> section (already filtered by score + autosave key above)
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
```

- [ ] **Step 2: Export formatter from pipeline module**

In `src/memory/pipeline/mod.rs`, add:

```rust
pub mod formatter;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check 2>&1 | head -20`

- [ ] **Step 4: Commit**

```bash
git add src/memory/pipeline/formatter.rs src/memory/pipeline/mod.rs
git commit -m "feat(memory): add Formatter for XML memory injection"
```

---

### Task 9: Pipeline Orchestrator

**Files:**
- Modify: `src/memory/pipeline/mod.rs`

- [ ] **Step 1: Implement Pipeline struct and process_turn**

Replace `src/memory/pipeline/mod.rs` entirely:

```rust
// src/memory/pipeline/mod.rs
pub mod config;
pub mod buffer;
pub mod store;
pub mod boundary;
pub mod extractor;
pub mod formatter;

pub use config::PipelineConfig;

use anyhow::Result;
use std::sync::Arc;
use tracing::{info, warn, error};

use crate::memory::traits::Memory;
use crate::providers::traits::Provider;

use self::boundary::{BoundaryDetector, FlushDecision};
use self::buffer::Buffer;
use self::extractor::episode::EpisodeExtractor;
use self::extractor::event_log::EventLogExtractor;
use self::formatter::Formatter;
use self::store::PipelineStore;

/// Memory Pipeline — boundary detection, extraction, and XML injection.
pub struct Pipeline {
    buffer: Buffer,
    boundary: BoundaryDetector,
    episode_extractor: EpisodeExtractor,
    event_log_extractor: EventLogExtractor,
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
            buffer: Buffer::new(conn),
            boundary: BoundaryDetector::new(provider.clone(), config.clone(), default_model.clone()),
            episode_extractor: EpisodeExtractor::new(provider.clone(), config.clone(), default_model.clone()),
            event_log_extractor: EventLogExtractor::new(provider, config.clone(), default_model),
            formatter: Formatter::new(store.clone(), memory, config.clone()),
            store,
            config,
        }
    }

    /// Append a message to the session buffer. Call synchronously before spawn.
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
    ) -> String {
        self.formatter
            .build_xml_context(user_query, min_relevance_score, session_id)
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

        // Save event logs (non-blocking: failure does NOT block buffer clear;
        // episode narrative is already persisted so clearing is safe).
        // Design decision: EventLog INSERT failure only loses the atomic facts
        // for this episode — the episode itself is preserved and the buffer is
        // cleared to avoid re-processing on the next turn.
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

        // Clear buffer — safe because episode was saved successfully above.
        // If event_logs failed, only atomic facts are lost; the narrative is intact.
        self.buffer.clear(session_id).await?;

        Ok(())
    }

    /// Check if pipeline is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check 2>&1 | head -20`

- [ ] **Step 3: Commit**

```bash
git add src/memory/pipeline/mod.rs
git commit -m "feat(memory): add Pipeline orchestrator with process_turn"
```

---

### Task 10: Integration — channels/mod.rs

**Files:**
- Modify: `src/channels/mod.rs`

This is the most delicate integration. The Pipeline needs to:
1. Be constructed during channel setup (where `Provider`, `Memory` are available)
2. `append_to_buffer(user_msg)` — on incoming message
3. `build_context()` — replace existing `build_memory_context()` call
4. `append_to_buffer(assistant_msg)` — after response
5. `process_turn()` — fire-and-forget via tokio::spawn

- [ ] **Step 1: Find where the channel runtime context is built**

Read `src/channels/mod.rs` around lines 250-400 to find where `ChannelRuntimeContext` is constructed and where `ctx.memory` is available. Note the exact location.

- [ ] **Step 2: Add Pipeline as an Option in the runtime context**

Add to `ChannelRuntimeContext` (or wherever the channel state is held):

```rust
pipeline: Option<Arc<crate::memory::pipeline::Pipeline>>,
```

- [ ] **Step 3: Construct Pipeline during channel setup**

Where `ctx.memory` and `ctx.provider` are initialized, add. The pipeline opens its **own** SQLite connection to the same `brain.db` file (separate from `SqliteMemory`'s connection — SQLite WAL mode supports concurrent readers/writers):

```rust
let pipeline = if config.memory.pipeline.enabled {
    // Open a second connection to brain.db for the pipeline
    let brain_path = workspace_dir.join("memory/brain.db");
    match rusqlite::Connection::open(&brain_path) {
        Ok(conn) => {
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
                .ok();
            let store = crate::memory::pipeline::store::PipelineStore::new(
                Arc::new(tokio::sync::Mutex::new(conn))
            );
            if let Err(e) = store.init_schema().await {
                warn!("Pipeline schema init failed: {e}");
            }
            let default_model = config.provider.model.clone();
            Some(Arc::new(crate::memory::pipeline::Pipeline::new(
                provider.clone(),
                memory.clone(),
                store,
                config.memory.pipeline.clone(),
                default_model,
            )))
        }
        Err(e) => {
            error!("Failed to open brain.db for pipeline: {e}");
            None
        }
    }
} else {
    None
};
```

**Note**: `workspace_dir` should be available in the channel context (it's used for IDENTITY.md, MEMORY.md, etc.). Find the exact variable name by searching for `workspace` in the channel setup code.

- [ ] **Step 4: Hook append_to_buffer for user message**

Near line 2510 (where auto-save happens), add:

```rust
if let Some(ref pipeline) = pipeline {
    let _ = pipeline.append_to_buffer(&history_key, "user", &msg.content).await;
}
```

- [ ] **Step 5: Replace build_memory_context with pipeline formatter**

At lines ~2622-2670, the existing code already recalls memory on **every turn** and appends it to the **system prompt** (not the user message). The pipeline replaces the `build_memory_context` call with XML-formatted output.

The existing block to replace (lines ~2622-2670):

```rust
let sender_memory_fut = build_memory_context(
    ctx.memory.as_ref(),
    &msg.content,
    ctx.min_relevance_score,
    Some(&msg.sender),
);
let (sender_memory, group_memory) = if is_group_chat {
    let group_memory_fut = build_memory_context(
        ctx.memory.as_ref(),
        &msg.content,
        ctx.min_relevance_score,
        Some(&history_key),
    );
    tokio::join!(sender_memory_fut, group_memory_fut)
} else {
    (sender_memory_fut.await, String::new())
};
// ...
let memory_context = if group_memory.is_empty() {
    sender_memory
} else if sender_memory.is_empty() {
    group_memory
} else {
    format!("{sender_memory}\n{group_memory}")
};
// ...
if !memory_context.is_empty() {
    let _ = write!(system_prompt, "\n\n{memory_context}");
}
```

Replace with:

```rust
let memory_context = if let Some(ref pipeline) = ctx.pipeline {
    // Pipeline: XML-formatted episodic + facts injection every turn
    pipeline.build_context(
        &msg.content,
        ctx.min_relevance_score,
        Some(&history_key),
    ).await
} else {
    // Legacy: [Memory context] flat-key block
    let sender_memory_fut = build_memory_context(
        ctx.memory.as_ref(),
        &msg.content,
        ctx.min_relevance_score,
        Some(&msg.sender),
    );
    let (sender_memory, group_memory) = if is_group_chat {
        let group_memory_fut = build_memory_context(
            ctx.memory.as_ref(),
            &msg.content,
            ctx.min_relevance_score,
            Some(&history_key),
        );
        tokio::join!(sender_memory_fut, group_memory_fut)
    } else {
        (sender_memory_fut.await, String::new())
    };
    if group_memory.is_empty() {
        sender_memory
    } else if sender_memory.is_empty() {
        group_memory
    } else {
        format!("{sender_memory}\n{group_memory}")
    }
};
if !memory_context.is_empty() {
    let _ = write!(system_prompt, "\n\n{memory_context}");
}
```

**Key differences from what the plan originally assumed:**
- Memory is injected into `system_prompt` (not prepended to user message `last_turn.content`)
- Memory is already recalled on every turn (no `had_prior_history` guard to change)
- Variable is `ctx.min_relevance_score` (not `runtime_defaults.min_relevance_score`)
- Group-chat dual-recall (sender + group) is preserved in the legacy path

- [ ] **Step 6: Hook append_to_buffer for assistant response + process_turn**

Near line 3172 (after the fire-and-forget memory consolidation `tokio::spawn`), add:

```rust
if let Some(ref pipeline) = pipeline {
    // Sync: append assistant response to buffer
    let _ = pipeline.append_to_buffer(&history_key, "assistant", &delivered_response).await;

    // Async: fire-and-forget post-turn processing
    let pipeline_clone = pipeline.clone();
    let session = history_key.clone();
    let uid = msg.sender.clone();
    tokio::spawn(async move {
        pipeline_clone.process_turn(&session, &uid).await;
    });
}
```

- [ ] **Step 7: Verify it compiles**

Run: `cargo check 2>&1 | head -20`
This step will likely need iteration — the exact variable names and code structure depend on the channel implementation.

- [ ] **Step 8: Commit**

```bash
git add src/channels/mod.rs
git commit -m "feat(memory): integrate Pipeline into channel message handling"
```

---

### Task 11: Integration — agent/loop_.rs

**Files:**
- Modify: `src/agent/loop_.rs`

**Note:** `src/agent/loop_.rs` is a single file (not a directory). There is no `context.rs` subdirectory — the `build_context` function lives directly in `loop_.rs` at line ~291.

The agent loop also builds memory context (for non-channel invocations like CLI). It should also use the XML formatter when pipeline is enabled.

- [ ] **Step 1: Add pipeline-aware context building**

In `src/agent/loop_.rs`, near the existing `build_context()` function (line ~291), add a new function:

```rust
/// Build context using the pipeline's XML formatter if available,
/// falling back to the standard [Memory context] format.
async fn build_context_with_pipeline(
    pipeline: Option<&crate::memory::pipeline::Pipeline>,
    mem: &dyn Memory,
    user_msg: &str,
    min_relevance_score: f64,
    session_id: Option<&str>,
) -> String {
    if let Some(pipeline) = pipeline {
        if pipeline.is_enabled() {
            return pipeline.build_context(user_msg, min_relevance_score, session_id).await;
        }
    }
    build_context(mem, user_msg, min_relevance_score, session_id).await
}
```

- [ ] **Step 2: Update call sites in loop_.rs**

In `src/agent/loop_.rs` there are three `build_context` call sites at lines **~3899**, **~4177**, and **~4726**. Replace each with `build_context_with_pipeline`, passing the pipeline reference. Example for the first call site:

```rust
// Before:
let mem_context = build_context(
    mem.as_ref(),
    &effective_msg,
    config.memory.min_relevance_score,
    memory_session_id.as_deref(),
).await;

// After:
let mem_context = build_context_with_pipeline(
    pipeline.as_deref(),
    mem.as_ref(),
    &effective_msg,
    config.memory.min_relevance_score,
    memory_session_id.as_deref(),
).await;
```

**Note**: `pipeline` needs to be threaded through to the agent loop — check how `config` and `mem` are passed and add the pipeline reference similarly. The CLI path does not require pipeline (pass `None`).

- [ ] **Step 3: Verify it compiles**

Run: `cargo check 2>&1 | head -20`

- [ ] **Step 4: Commit**

```bash
git add src/agent/loop_.rs
git commit -m "feat(memory): integrate XML injection into agent loop context"
```

---

### Task 12: Build & Test

- [ ] **Step 1: Full build check**

Run: `cargo build 2>&1 | tail -20`
Expected: successful build

- [ ] **Step 2: Run existing tests**

Run: `cargo test 2>&1 | tail -30`
Expected: all existing tests pass (no regressions)

- [ ] **Step 3: Test config deserialization**

Manually test that a config with `[memory.pipeline]` section parses correctly:

```bash
cargo test -- --test-threads=1 config 2>&1 | tail -20
```

- [ ] **Step 4: Cross-compile for aarch64 (CI)**

Push to nexus branch and verify GitHub Actions builds successfully:

```bash
git push origin nexus
```

Check CI at `https://github.com/neokree/zeroclaw/actions`

- [ ] **Step 5: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: resolve build issues for memory pipeline Phase 1"
```

---

### Task 13: Deploy & Validate

- [ ] **Step 1: Download CI artifact**

After CI passes, download the aarch64 binary from GitHub Actions.

- [ ] **Step 2: Update config on device**

SSH to Bison Pro and add pipeline config:

```bash
ssh bison-pro "cat >> ~/.zeroclaw/config.toml << 'EOF'

[memory.pipeline]
enabled = true
extraction_model = "moonshotai/kimi-k2.5"
prompt_language = "it"
boundary_threshold = 0.6
hard_token_limit = 4096
hard_message_limit = 30
buffer_ttl_hours = 24
max_episodic_entries = 5
max_facts_entries = 5
EOF"
```

- [ ] **Step 3: Backup brain.db**

```bash
ssh bison-pro "cp ~/workspace/memory/brain.db ~/workspace/memory/brain_backup_20260405.db"
```

- [ ] **Step 4: Deploy binary and restart**

```bash
scp zeroclaw-aarch64 bison-pro:~/.local/bin/zeroclaw
ssh bison-pro "pm2 restart zeroclaw-gateway --update-env"
```

- [ ] **Step 5: Validate via Telegram**

Send a few test messages via Telegram. Check logs for pipeline activity:

```bash
ssh bison-pro "pm2 logs zeroclaw-gateway --lines 50 --nostream" | grep pipeline
```

- [ ] **Step 6: Verify tables were created**

```bash
ssh bison-pro "python3 -c \"
import sqlite3
conn = sqlite3.connect('/home/nexus/workspace/memory/brain.db')
tables = conn.execute(\\\"SELECT name FROM sqlite_master WHERE type='table'\\\").fetchall()
print([t[0] for t in tables])
\""
```

Expected: should include `memcell_buffer`, `episodes`, `event_logs` alongside existing tables.

- [ ] **Step 7: Commit config backup note**

```bash
git add -A
git commit -m "docs: note Phase 1 deployment to Bison Pro"
```
