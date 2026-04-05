# Memory Pipeline Phase 2 — Foresight Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Foresight extraction (behavioral predictions with time bounds) to the existing Memory Pipeline, with XML injection of active foresights.

**Architecture:** Extends `src/memory/pipeline/` (built in Phase 1) with a new `ForesightExtractor` that runs sequentially after Episode extraction, receives the episode narrative as context, and produces time-bounded predictions stored in a new `foresights` table. The `Formatter` is updated to include a `<foresight>` section in the XML block. Foresight runs as a non-blocking step — if it fails, episode + event_log are still saved.

**Tech Stack:** Rust, SQLite (rusqlite), tokio, serde_json, chrono

**Spec:** `docs/superpowers/specs/2026-04-05-zeroclaw-memory-pipeline-design.md` (Phase 2 section)

**Prerequisite:** Phase 1 must be implemented (`src/memory/pipeline/` module with buffer, boundary, episode + event_log extractors, store, formatter, config).

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/memory/pipeline/extractor/foresight.rs` | `ForesightExtractor`: LLM call → `Vec<ForesightEntry>` |
| `src/memory/pipeline/prompts/en/foresight.txt` | English foresight extraction prompt |
| `src/memory/pipeline/prompts/it/foresight.txt` | Italian foresight extraction prompt |

### Modified Files

| File | Change |
|------|--------|
| `src/memory/pipeline/store.rs` | Add `foresights` table creation + `save_foresights()` + `get_active_foresights()` methods |
| `src/memory/pipeline/config.rs` | Add `max_foresight_entries: usize` field (default 3) |
| `src/memory/pipeline/extractor/mod.rs` | Add `pub mod foresight;`, update orchestration to run foresight after episode |
| `src/memory/pipeline/formatter.rs` | Add `<foresight>` section to XML block |
| `src/memory/pipeline/mod.rs` | Update `process_turn()` to pass episode result to foresight extractor |

---

## Chunk 1: Store + Config + ForesightExtractor

### Task 1: Add `foresights` table to PipelineStore

**Files:**
- Modify: `src/memory/pipeline/store.rs`

- [ ] **Step 1: Add foresights table creation to `init_schema()`**

Add after the `event_logs` table creation:

```rust
conn.execute_batch(
    "CREATE TABLE IF NOT EXISTS foresights (
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
    CREATE INDEX IF NOT EXISTS idx_foresights_end_time ON foresights(end_time);",
)?;
```

- [ ] **Step 2: Add `ForesightRow` struct and `save_foresights()` method**

```rust
pub struct ForesightRow {
    pub id: String,
    pub episode_id: String,
    pub user_id: Option<String>,
    pub content: String,
    pub evidence: String,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub created_at: String,
    // embedding: BLOB omitted — will be populated when embedding pipeline is added
}

pub async fn save_foresights(&self, foresights: &[ForesightRow]) -> Result<()> {
    let conn = self.conn.lock().await;
    let mut stmt = conn.prepare_cached(
        "INSERT INTO foresights (id, episode_id, user_id, content, evidence, start_time, end_time, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    for f in foresights {
        stmt.execute(params![
            f.id, f.episode_id, f.user_id, f.content, f.evidence,
            f.start_time, f.end_time, f.created_at,
        ])?;
    }
    Ok(())
}
```

- [ ] **Step 3: Add `get_active_foresights()` method**

Returns foresights where `end_time >= today` (or `end_time IS NULL`), ordered by `created_at DESC`, limited to N.

**Note**: The spec calls for ordering by embedding similarity to the user message. Since Phase 1 deferred embedding generation (YAGNI), we use `created_at DESC` as a temporary fallback. TODO: switch to embedding-based ranking when the embedding pipeline is added.

```rust
// TODO: replace created_at ordering with embedding similarity when available
pub async fn get_active_foresights(
    &self,
    user_id: Option<&str>,
    limit: usize,
) -> Result<Vec<ForesightRow>> {
    let conn = self.conn.lock().await;
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    // Filter by user_id when provided — prevents mixing foresights across
    // different channel senders in multi-channel deployments.
    let rows: Vec<ForesightRow> = if let Some(uid) = user_id {
        let mut stmt = conn.prepare_cached(
            "SELECT id, episode_id, user_id, content, evidence, start_time, end_time, created_at
             FROM foresights
             WHERE (end_time IS NULL OR end_time >= ?1)
               AND user_id = ?2
             ORDER BY created_at DESC
             LIMIT ?3",
        )?;
        stmt.query_map(params![today, uid, limit as i64], |row| {
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
             WHERE end_time IS NULL OR end_time >= ?1
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;
        stmt.query_map(params![today, limit as i64], |row| {
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
```

- [ ] **Step 4: Compile and verify**

Run: `cargo check`
Expected: compiles with no errors.

- [ ] **Step 5: Commit**

```bash
git add src/memory/pipeline/store.rs
git commit -m "feat(pipeline): add foresights table and store methods"
```

---

### Task 2: Add `max_foresight_entries` to PipelineConfig

**Files:**
- Modify: `src/memory/pipeline/config.rs`

- [ ] **Step 1: Add field to `PipelineConfig`**

```rust
pub max_foresight_entries: usize,
```

- [ ] **Step 2: Add default value**

In `Default` impl:
```rust
max_foresight_entries: 3,
```

- [ ] **Step 3: Compile and verify**

Run: `cargo check`

- [ ] **Step 4: Commit**

```bash
git add src/memory/pipeline/config.rs
git commit -m "feat(pipeline): add max_foresight_entries config (default 3)"
```

---

### Task 3: Create foresight prompts

**Files:**
- Create: `src/memory/pipeline/prompts/en/foresight.txt`
- Create: `src/memory/pipeline/prompts/it/foresight.txt`

- [ ] **Step 1: Create English foresight prompt**

```text
Based on this conversation and its episode summary, generate 4-8
predictions about how the user's behavior or situation may change.

Episode summary:
{episode_summary}

Original conversation:
{buffer_content}

Use "associative prediction": predict personal-level behavioral
impacts, not summaries.

Each prediction must be:
- Specific, reasonable, and verifiable
- Max 40 words
- Grounded in evidence from the conversation
- Time-bounded (when does it start, when does it expire?)

Respond ONLY with valid JSON:
{"foresights": [{"content": "prediction text (max 40 words)", "evidence": "supporting fact from conversation (max 40 words)", "start_time": "YYYY-MM-DD", "end_time": "YYYY-MM-DD"}]}

If no meaningful predictions can be made, return {"foresights": []}
```

- [ ] **Step 2: Create Italian foresight prompt**

Same structure, translated to Italian. Key translations:
- "predictions" → "previsioni"
- "behavioral impacts" → "impatti comportamentali"
- "associative prediction" → "previsione associativa"
- "evidence" → "evidenza"

- [ ] **Step 3: Commit**

```bash
git add src/memory/pipeline/prompts/en/foresight.txt src/memory/pipeline/prompts/it/foresight.txt
git commit -m "feat(pipeline): add foresight extraction prompts (en, it)"
```

---

### Task 4: Create ForesightExtractor

**Files:**
- Create: `src/memory/pipeline/extractor/foresight.rs`
- Modify: `src/memory/pipeline/extractor/mod.rs`

- [ ] **Step 1: Define ForesightEntry and response structs**

```rust
// src/memory/pipeline/extractor/foresight.rs
use anyhow::Result;
use serde::Deserialize;

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
```

- [ ] **Step 2: Implement `extract()` function**

The function takes the buffer content AND the episode summary (from EpisodeExtractor output). It calls the LLM and parses the JSON response.

```rust
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};

const PROMPT_EN: &str = include_str!("../prompts/en/foresight.txt");
const PROMPT_IT: &str = include_str!("../prompts/it/foresight.txt");

pub async fn extract(
    provider: &dyn Provider,
    model: &str,
    buffer_content: &str,
    episode_summary: &str,
    prompt_language: &str,
) -> Result<Vec<ForesightEntry>> {
    let template = match prompt_language {
        "it" => PROMPT_IT,
        _ => PROMPT_EN,
    };

    let prompt = template
        .replace("{episode_summary}", episode_summary)
        .replace("{buffer_content}", buffer_content);

    let messages = [
        ChatMessage::system("You are a memory foresight engine. Extract behavioral predictions."),
        ChatMessage::user(&prompt),
    ];

    let request = ChatRequest {
        messages: &messages,
        tools: None,
    };

    let response = provider.chat(request, model, 0.1).await?;
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
```

**Note**: Uses the same `ChatRequest` + `ChatResponse.text` pattern as `EpisodeExtractor` and `EventLogExtractor` in Phase 1.

- [ ] **Step 3: Add `pub mod foresight;` to extractor/mod.rs**

```rust
pub mod foresight;
```

- [ ] **Step 4: Compile and verify**

Run: `cargo check`

- [ ] **Step 5: Commit**

```bash
git add src/memory/pipeline/extractor/foresight.rs src/memory/pipeline/extractor/mod.rs
git commit -m "feat(pipeline): add ForesightExtractor with LLM extraction"
```

---

## Chunk 2: Orchestration + Formatter Integration

### Task 5: Update extraction orchestration to include foresight

**Files:**
- Modify: `src/memory/pipeline/extractor/mod.rs`
- Modify: `src/memory/pipeline/mod.rs`

The current Phase 1 flow runs Episode + EventLog in parallel. Phase 2 adds Foresight as a sequential step after Episode completes.

- [ ] **Step 1: Update `ExtractionResult` to include foresights**

Add to the existing `ExtractionResult` struct in `extractor/mod.rs`:

```rust
pub foresights: Vec<foresight::ForesightEntry>,
```

- [ ] **Step 2: Update extraction orchestration**

In the extraction orchestration function (in `extractor/mod.rs` or `mod.rs` depending on Phase 1 structure), change the flow from:

```rust
// Phase 1: parallel
let (episode, event_logs) = tokio::join!(
    episode::extract(...),
    event_log::extract(...),
);
```

To:

```rust
// Phase 2: episode + event_log parallel, then foresight sequential
let (episode_result, event_log_result) = tokio::join!(
    episode::extract(provider, model, buffer_content, prompt_language),
    event_log::extract(provider, model, buffer_content, prompt_language),
);

let episode = episode_result?;
let event_logs = event_log_result?;

// Foresight runs after episode (needs episode summary as context)
let foresights = match foresight::extract(
    provider,
    model,
    buffer_content,
    &episode.summary,
    prompt_language,
).await {
    Ok(f) => f,
    Err(e) => {
        tracing::warn!("Foresight extraction failed (non-blocking): {e}");
        vec![]
    }
};
```

**Key**: Foresight failure is **non-blocking** — it logs a warning and returns an empty vec. Episode + EventLog are still saved.

- [ ] **Step 3: Update `process_turn()` in `mod.rs` to save foresights**

After the existing save for episodes + event_logs, add:

```rust
if !result.foresights.is_empty() {
    let foresight_rows: Vec<ForesightRow> = result.foresights.iter().map(|f| {
        ForesightRow {
            id: uuid::Uuid::new_v4().to_string(),
            episode_id: episode_id.clone(),
            user_id: user_id.map(|s| s.to_string()),
            content: f.content.clone(),
            evidence: f.evidence.clone(),
            start_time: f.start_time.clone(),
            end_time: f.end_time.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }).collect();

    if let Err(e) = self.store.save_foresights(&foresight_rows).await {
        tracing::warn!("Failed to save foresights (non-blocking): {e}");
    }
}
```

- [ ] **Step 4: Compile and verify**

Run: `cargo check`

- [ ] **Step 5: Commit**

```bash
git add src/memory/pipeline/extractor/mod.rs src/memory/pipeline/mod.rs
git commit -m "feat(pipeline): integrate foresight into extraction orchestration"
```

---

### Task 6: Update Formatter to include `<foresight>` section

**Files:**
- Modify: `src/memory/pipeline/formatter.rs`

- [ ] **Step 1: Add foresight retrieval to `build_xml_context()`**

After the existing `<episodic>` and `<facts>` sections, add:

```rust
// Foresight section — active (non-expired) predictions, filtered by user_id
// to avoid mixing foresights across different channel senders.
let foresights = store.get_active_foresights(user_id, config.max_foresight_entries).await?;
if !foresights.is_empty() {
    xml.push_str("  <foresight>\n");
    for f in &foresights {
        xml.push_str(&format!("    - {}\n", f.content));
    }
    xml.push_str("  </foresight>\n");
}
```

The `<foresight>` section is placed after `<facts>` and before the closing `</memory>` tag.

- [ ] **Step 2: Update the `build_xml_context()` signature**

The Phase 2 `build_xml_context()` now takes an additional `user_id: Option<&str>` parameter (added in Fix 3 above). Update all call sites in `Pipeline::build_context()` in `mod.rs` to pass the user_id. The `user_id` is available as `msg.sender` in the channels context.

- [ ] **Step 3: Compile and verify**

Run: `cargo check`

- [ ] **Step 4: Verify XML output format**

The expected XML output with foresight should look like:

```text
<memory>
  <episodic>
    [1] (2026-04-05) ...
  </episodic>
  <facts>
    - ...
  </facts>
  <foresight>
    - User will likely want to start Rust implementation this week
    - User may need to test boundary detection on real conversations
  </foresight>
</memory>
```

- [ ] **Step 5: Commit**

```bash
git add src/memory/pipeline/formatter.rs
git commit -m "feat(pipeline): add <foresight> section to XML injection"
```

---

### Task 7: Add `max_foresight_entries` to config schema

**Files:**
- Modify: `src/config/schema.rs`

- [ ] **Step 1: Add `max_foresight_entries` to pipeline config in schema.rs**

In the section where Phase 1 added pipeline config fields to `MemoryConfig` (or its nested pipeline struct), add:

```rust
/// Maximum foresight entries to show in XML injection (Phase 2)
pub max_foresight_entries: Option<usize>,
```

Wire it to `PipelineConfig::max_foresight_entries` during config loading, defaulting to 3 if absent.

- [ ] **Step 2: Compile and verify**

Run: `cargo check`

- [ ] **Step 3: Commit**

```bash
git add src/config/schema.rs
git commit -m "feat(config): add max_foresight_entries to pipeline config"
```

---

### Task 8: Integration test

**Files:**
- No new test file — test manually via Telegram or a local script

- [ ] **Step 1: Build the binary**

Run: `cargo build --release`

- [ ] **Step 2: Verify schema migration**

Start ZeroClaw with `[memory.pipeline] enabled = true` in config. Check that `foresights` table exists:

```bash
ssh bison-pro "python3 -c \"
import sqlite3
conn = sqlite3.connect('/home/nexus/workspace/memory/brain.db')
cursor = conn.execute(\\\"SELECT name FROM sqlite_master WHERE type='table' AND name='foresights'\\\")
print('foresights table:', 'EXISTS' if cursor.fetchone() else 'MISSING')
conn.close()
\""
```

- [ ] **Step 3: Test foresight extraction**

Send a few messages via Telegram that form a complete conversation segment (trigger a boundary flush). Check logs for foresight extraction:

```bash
ssh bison-pro "pm2 logs zeroclaw-gateway --lines 50 --nostream | grep -i foresight"
```

- [ ] **Step 4: Verify foresight storage**

```bash
ssh bison-pro "python3 -c \"
import sqlite3
conn = sqlite3.connect('/home/nexus/workspace/memory/brain.db')
for row in conn.execute('SELECT id, content, evidence, start_time, end_time FROM foresights LIMIT 5'):
    print(row)
conn.close()
\""
```

- [ ] **Step 5: Verify XML injection includes foresight**

Check that the `<foresight>` section appears in the injected memory context. Look in logs for the XML block or add a temporary debug log in `formatter.rs`.

- [ ] **Step 6: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix(pipeline): foresight integration fixes from testing"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `store.rs` | Add `foresights` table + `save_foresights()` + `get_active_foresights()` |
| 2 | `config.rs` | Add `max_foresight_entries` (default 3) |
| 3 | `prompts/en/foresight.txt`, `prompts/it/foresight.txt` | Foresight extraction prompts |
| 4 | `extractor/foresight.rs`, `extractor/mod.rs` | `ForesightExtractor` with LLM call + JSON parse |
| 5 | `extractor/mod.rs`, `mod.rs` | Orchestration: foresight runs sequentially after episode |
| 6 | `formatter.rs` | Add `<foresight>` section to XML block |
| 7 | `config/schema.rs` | Wire `max_foresight_entries` to top-level config |
| 8 | — | Integration test on device |
