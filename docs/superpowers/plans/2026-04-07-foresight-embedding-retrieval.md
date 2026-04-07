# Foresight Embedding Similarity Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ORDER BY created_at DESC` foresight retrieval with cosine similarity ranking against the user's current query, with graceful fallback to recency when the embedder is unavailable.

**Architecture:** `EmbeddingProvider` is constructed in `channels/mod.rs` from `config.memory` fields and injected into `Pipeline`. At save time, foresight content is embedded in a single batch call and stored as `BLOB`. At retrieval time, the query is embedded via `embed_one`, active foresights are ranked by cosine similarity in Rust, and top-N are returned. Every step falls back gracefully when the embedder is absent or fails.

**Tech Stack:** Rust, SQLite (`rusqlite`), `crate::memory::embeddings` (existing), `crate::memory::vector` (existing).

---

## Chunk 1: Store — ForesightRow + save_foresights + get_active_foresights

### Task 1: Add `embedding` field to `ForesightRow` and update `save_foresights`

**Files:**
- Modify: `src/memory/pipeline/store.rs`

- [ ] **Step 1: Write failing test for `save_foresights` with embedding**

Add inside a new `#[cfg(test)] mod tests` block at the bottom of `store.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    async fn make_store() -> PipelineStore {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        let store = PipelineStore::new(Arc::new(Mutex::new(conn)));
        store.init_schema().await.unwrap();
        store
    }

    #[tokio::test]
    async fn test_save_foresight_with_embedding() {
        let store = make_store().await;
        let embedding = vec![0.1f32, 0.2, 0.3];
        let blob = crate::memory::vector::vec_to_bytes(&embedding);
        let row = ForesightRow {
            id: "f1".to_string(),
            episode_id: None,
            user_id: Some("u1".to_string()),
            content: "user will code tomorrow".to_string(),
            evidence: "user mentioned coding plans".to_string(),
            start_time: None,
            end_time: Some("2099-12-31".to_string()),
            created_at: "2026-04-07T00:00:00Z".to_string(),
            embedding: Some(blob),
        };
        store.save_foresights(&[row]).await.unwrap();
    }

    #[tokio::test]
    async fn test_save_foresight_without_embedding() {
        let store = make_store().await;
        let row = ForesightRow {
            id: "f2".to_string(),
            episode_id: None,
            user_id: Some("u1".to_string()),
            content: "user will rest".to_string(),
            evidence: "user mentioned rest".to_string(),
            start_time: None,
            end_time: Some("2099-12-31".to_string()),
            created_at: "2026-04-07T00:00:00Z".to_string(),
            embedding: None,
        };
        store.save_foresights(&[row]).await.unwrap();
    }
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/fabiobiola/Developer/Nexus/zeroclaw-source
cargo test -p zeroclaw memory::pipeline::store::tests -- --nocapture 2>&1 | tail -20
```

Expected: compile error — `ForesightRow` has no field `embedding`.

- [ ] **Step 3: Add `embedding: Option<Vec<u8>>` to `ForesightRow`**

In `store.rs`, find `ForesightRow` struct (around line 350) and replace:

```rust
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
```

with:

```rust
pub struct ForesightRow {
    pub id: String,
    pub episode_id: Option<String>,
    pub user_id: Option<String>,
    pub content: String,
    pub evidence: String,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub created_at: String,
    pub embedding: Option<Vec<u8>>,
}
```

- [ ] **Step 4: Update `save_foresights` SQL to 9 columns**

Find `save_foresights` (around line 171) and replace the INSERT statement and `.execute()` call:

```rust
pub async fn save_foresights(&self, foresights: &[ForesightRow]) -> Result<()> {
    let conn = self.conn.lock().await;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO foresights (id, episode_id, user_id, content, evidence, start_time, end_time, created_at, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        for f in foresights {
            stmt.execute(rusqlite::params![
                f.id, f.episode_id, f.user_id, f.content, f.evidence,
                f.start_time, f.end_time, f.created_at, f.embedding,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cargo test -p zeroclaw memory::pipeline::store::tests::test_save_foresight -- --nocapture 2>&1 | tail -20
```

Expected: both `test_save_foresight_with_embedding` and `test_save_foresight_without_embedding` PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memory/pipeline/store.rs
git commit -m "feat(pipeline): add embedding field to ForesightRow and save_foresights"
```

---

### Task 2: Update `get_active_foresights` to rank by cosine similarity

**Files:**
- Modify: `src/memory/pipeline/store.rs`

- [ ] **Step 1: Write failing test for similarity retrieval**

Add to the `tests` block in `store.rs`:

```rust
#[tokio::test]
async fn test_get_active_foresights_by_similarity() {
    let store = make_store().await;

    // Insert two foresights with embeddings
    // f_relevant: embedding close to query [1.0, 0.0, 0.0]
    // f_distant:  embedding orthogonal to query
    let f_relevant = ForesightRow {
        id: "rel".to_string(),
        episode_id: None,
        user_id: Some("u1".to_string()),
        content: "user will code in Rust".to_string(),
        evidence: "".to_string(),
        start_time: None,
        end_time: Some("2099-12-31".to_string()),
        created_at: "2026-04-07T00:00:00Z".to_string(),
        embedding: Some(crate::memory::vector::vec_to_bytes(&[1.0f32, 0.1, 0.0])),
    };
    let f_distant = ForesightRow {
        id: "dist".to_string(),
        episode_id: None,
        user_id: Some("u1".to_string()),
        content: "user will sleep early".to_string(),
        evidence: "".to_string(),
        start_time: None,
        end_time: Some("2099-12-31".to_string()),
        created_at: "2026-04-07T01:00:00Z".to_string(), // more recent, but less relevant
        embedding: Some(crate::memory::vector::vec_to_bytes(&[0.0f32, 1.0, 0.0])),
    };
    store.save_foresights(&[f_relevant, f_distant]).await.unwrap();

    // Query embedding close to f_relevant
    let query_emb = vec![1.0f32, 0.0, 0.0];
    let results = store
        .get_active_foresights(Some("u1"), 2, Some(query_emb))
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    // Most relevant first
    assert_eq!(results[0].id, "rel");
    assert_eq!(results[1].id, "dist");
}

#[tokio::test]
async fn test_get_active_foresights_fallback_to_recency() {
    let store = make_store().await;

    let f1 = ForesightRow {
        id: "old".to_string(),
        episode_id: None,
        user_id: Some("u1".to_string()),
        content: "older foresight".to_string(),
        evidence: "".to_string(),
        start_time: None,
        end_time: Some("2099-12-31".to_string()),
        created_at: "2026-04-07T00:00:00Z".to_string(),
        embedding: None,
    };
    let f2 = ForesightRow {
        id: "new".to_string(),
        episode_id: None,
        user_id: Some("u1".to_string()),
        content: "newer foresight".to_string(),
        evidence: "".to_string(),
        start_time: None,
        end_time: Some("2099-12-31".to_string()),
        created_at: "2026-04-07T01:00:00Z".to_string(),
        embedding: None,
    };
    store.save_foresights(&[f1, f2]).await.unwrap();

    // No query embedding → recency fallback
    let results = store
        .get_active_foresights(Some("u1"), 2, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "new"); // most recent first
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test -p zeroclaw memory::pipeline::store::tests::test_get_active_foresights -- --nocapture 2>&1 | tail -20
```

Expected: compile error — `get_active_foresights` does not yet accept a 3rd argument.

- [ ] **Step 3: Rewrite `get_active_foresights`**

Replace the entire `get_active_foresights` function (from the `// TODO: replace...` comment through its closing `}`):

```rust
pub async fn get_active_foresights(
    &self,
    user_id: Option<&str>,
    limit: usize,
    query_embedding: Option<Vec<f32>>,
) -> Result<Vec<ForesightRow>> {
    let conn = self.conn.lock().await;
    let today = Utc::now().format("%Y-%m-%d").to_string();

    let rows: Vec<ForesightRow> = if let Some(uid) = user_id {
        let mut stmt = conn.prepare_cached(
            "SELECT id, episode_id, user_id, content, evidence, start_time, end_time, created_at, embedding
             FROM foresights
             WHERE (end_time IS NULL OR end_time >= ?1)
               AND user_id = ?2
             ORDER BY created_at DESC",
        )?;
        stmt.query_map(rusqlite::params![today, uid], |row| {
            Ok(ForesightRow {
                id: row.get(0)?,
                episode_id: row.get(1)?,
                user_id: row.get(2)?,
                content: row.get(3)?,
                evidence: row.get(4)?,
                start_time: row.get(5)?,
                end_time: row.get(6)?,
                created_at: row.get(7)?,
                embedding: row.get(8)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?
    } else {
        let mut stmt = conn.prepare_cached(
            "SELECT id, episode_id, user_id, content, evidence, start_time, end_time, created_at, embedding
             FROM foresights
             WHERE (end_time IS NULL OR end_time >= ?1)
             ORDER BY created_at DESC",
        )?;
        stmt.query_map(rusqlite::params![today], |row| {
            Ok(ForesightRow {
                id: row.get(0)?,
                episode_id: row.get(1)?,
                user_id: row.get(2)?,
                content: row.get(3)?,
                evidence: row.get(4)?,
                start_time: row.get(5)?,
                end_time: row.get(6)?,
                created_at: row.get(7)?,
                embedding: row.get(8)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?
    };

    // If no query embedding → return as-is (already ordered by recency)
    let Some(qvec) = query_embedding else {
        return Ok(rows.into_iter().take(limit).collect());
    };

    // Rank by cosine similarity; rows without embedding get score -1.0
    let mut scored: Vec<(f32, ForesightRow)> = rows
        .into_iter()
        .map(|row| {
            let score = row
                .embedding
                .as_deref()
                .map(crate::memory::vector::bytes_to_vec)
                .map(|fvec| crate::memory::vector::cosine_similarity(&qvec, &fvec))
                .unwrap_or(-1.0);
            (score, row)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored.into_iter().take(limit).map(|(_, row)| row).collect())
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p zeroclaw memory::pipeline::store::tests::test_get_active_foresights -- --nocapture 2>&1 | tail -20
```

Expected: both tests PASS.

- [ ] **Step 5: Run full pipeline tests to check for regressions**

```bash
cargo test -p zeroclaw memory::pipeline -- --nocapture 2>&1 | tail -30
```

Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memory/pipeline/store.rs
git commit -m "feat(pipeline): rank foresights by cosine similarity with recency fallback"
```

---

## Chunk 2: Pipeline + Formatter — embedder injection and query embedding

### Task 3: Add embedder to `Pipeline` and embed foresights at save time

**Files:**
- Modify: `src/memory/pipeline/mod.rs`

- [ ] **Step 1: Add `embedder` field to `Pipeline` struct**

At the top of `mod.rs`, add the import after the existing `use` block:

```rust
use crate::memory::embeddings::EmbeddingProvider;
```

In the `Pipeline` struct, add the field after `config`:

```rust
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
    embedder: Option<Arc<dyn EmbeddingProvider>>,
}
```

- [ ] **Step 2: Update `Pipeline::new()` to accept embedder**

Add `embedder: Option<Arc<dyn EmbeddingProvider>>` as the last parameter and assign it:

```rust
pub fn new(
    provider: Arc<dyn Provider>,
    memory: Arc<dyn Memory>,
    store: PipelineStore,
    config: PipelineConfig,
    default_model: String,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
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
        embedder,
    }
}
```

- [ ] **Step 3: Embed foresights at save time in `process_turn_inner`**

Find the section that saves foresights (around line 186). The existing code has a guard:
```rust
if !foresight_entries.is_empty() {
    use crate::memory::pipeline::store::ForesightRow;
    let foresight_rows: Vec<ForesightRow> = foresight_entries.iter().map(|f| ForesightRow {
        ...
    }).collect();
    ...
}
```

**Replace the entire body inside the `if !foresight_entries.is_empty()` block** (keep the guard itself) with:

```rust
// Embed foresight content in a single batch call (fire-and-forget on failure)
let foresight_embeddings: Vec<Option<Vec<u8>>> = if let Some(emb) = &self.embedder {
    let contents: Vec<&str> = foresight_entries.iter().map(|f| f.content.as_str()).collect();
    match emb.embed(&contents).await {
        Ok(vecs) => vecs
            .into_iter()
            .map(|v| Some(crate::memory::vector::vec_to_bytes(&v)))
            .collect(),
        Err(e) => {
            tracing::warn!(
                target: "memory::pipeline",
                "Foresight embedding failed (saving with NULL BLOB): {e}"
            );
            vec![None; foresight_entries.len()]
        }
    }
} else {
    vec![None; foresight_entries.len()]
};

let foresight_rows: Vec<ForesightRow> = foresight_entries
    .iter()
    .zip(foresight_embeddings.into_iter())
    .map(|(f, emb)| ForesightRow {
        id: uuid::Uuid::new_v4().to_string(),
        episode_id: Some(saved.id.clone()),
        user_id: Some(user_id.to_string()),
        content: f.content.clone(),
        evidence: f.evidence.clone(),
        start_time: f.start_time.clone(),
        end_time: f.end_time.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        embedding: emb,
    })
    .collect();
```

- [ ] **Step 4: Update `build_context` to generate query embedding and pass it to formatter**

Replace the current `build_context` function:

```rust
pub async fn build_context(
    &self,
    user_query: &str,
    min_relevance_score: f64,
    session_id: Option<&str>,
    user_id: Option<&str>,
) -> String {
    let query_embedding = if let Some(emb) = &self.embedder {
        emb.embed_one(user_query).await.ok()
    } else {
        None
    };

    self.formatter
        .build_xml_context(user_query, min_relevance_score, session_id, user_id, query_embedding)
        .await
        .unwrap_or_default()
}
```

- [ ] **Step 5: Build to verify no compile errors**

```bash
cargo build -p zeroclaw 2>&1 | grep "^error" | head -20
```

Expected: errors only in `formatter.rs` (signature mismatch — fixed in Task 4) and `channels/mod.rs` (missing arg — fixed in Task 5).

- [ ] **Step 6: Commit (even if not yet compiling — WIP commit)**

```bash
git add src/memory/pipeline/mod.rs
git commit -m "feat(pipeline): inject EmbeddingProvider, embed foresights at save time"
```

---

### Task 4: Update `Formatter` to accept and pass `query_embedding`

**Files:**
- Modify: `src/memory/pipeline/formatter.rs`

- [ ] **Step 1: Update `build_xml_context` signature**

Add `query_embedding: Option<Vec<f32>>` as 5th parameter and pass it to `get_active_foresights`:

```rust
pub async fn build_xml_context(
    &self,
    user_query: &str,
    min_relevance_score: f64,
    session_id: Option<&str>,
    user_id: Option<&str>,
    query_embedding: Option<Vec<f32>>,
) -> Result<String> {
    // Fetch episodic memories (recency-based — unchanged)
    let episodes = self.store
        .recent_episodes(self.config.max_episodic_entries)
        .await
        .unwrap_or_default();

    // Fetch facts via existing Memory::recall (relevance-based — unchanged)
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

    // Fetch foresights — similarity-ranked if query_embedding available, recency fallback
    let foresights = self.store
        .get_active_foresights(user_id, self.config.max_foresight_entries, query_embedding)
        .await
        .unwrap_or_default();

    // ... rest of the function unchanged (profile, XML building)
```

Keep the rest of the function body (profile lookup, XML construction) exactly as-is.

- [ ] **Step 2: Build to verify formatter compiles**

```bash
cargo build -p zeroclaw 2>&1 | grep "^error" | head -20
```

Expected: only `channels/mod.rs` error remaining (Pipeline::new missing arg).

- [ ] **Step 3: Commit**

```bash
git add src/memory/pipeline/formatter.rs
git commit -m "feat(pipeline): pass query_embedding to foresight retrieval in formatter"
```

---

## Chunk 3: Channel wiring + final verification

### Task 5: Construct `EmbeddingProvider` in `channels/mod.rs` and pass to `Pipeline::new`

**Files:**
- Modify: `src/channels/mod.rs`

- [ ] **Step 1: Find the `Pipeline::new` call site**

Search for `Pipeline::new` in `channels/mod.rs` (around line 5578). The current call is:

```rust
Some(Arc::new(crate::memory::pipeline::Pipeline::new(
    Arc::clone(&provider),
    Arc::clone(&mem),
    store,
    config.memory.pipeline.clone(),
    default_model,
)))
```

- [ ] **Step 2: Add embedder construction and pass to `Pipeline::new`**

Replace with:

```rust
// Build embedding provider for pipeline (same config fields as build_sqlite_memory)
let pipeline_embedder: Option<Arc<dyn crate::memory::embeddings::EmbeddingProvider>> = {
    let ep = config.memory.embedding_provider.trim();
    if ep.is_empty() || ep == "none" {
        None
    } else {
        Some(Arc::from(crate::memory::embeddings::create_embedding_provider(
            ep,
            None, // no API key needed for custom/local provider
            config.memory.embedding_model.trim(),
            config.memory.embedding_dimensions,
        )))
    }
};

Some(Arc::new(crate::memory::pipeline::Pipeline::new(
    Arc::clone(&provider),
    Arc::clone(&mem),
    store,
    config.memory.pipeline.clone(),
    default_model,
    pipeline_embedder,
)))
```

- [ ] **Step 3: Full build — must compile clean**

```bash
cargo build -p zeroclaw 2>&1 | grep "^error" | head -20
```

Expected: zero errors.

- [ ] **Step 4: Run all pipeline tests**

```bash
cargo test -p zeroclaw memory::pipeline -- --nocapture 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
cargo test -p zeroclaw 2>&1 | tail -20
```

Expected: no new failures (pre-existing failures are acceptable, no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/channels/mod.rs
git commit -m "feat(pipeline): wire EmbeddingProvider into Pipeline at channel init"
```

---

### Task 6: Integration smoke test

- [ ] **Step 1: Push and trigger CI release**

The aarch64 cross-compile requires `aarch64-linux-gnu-gcc` which is not available on macOS without a cross-toolchain. Verification happens via CI.

```bash
git push origin nexus
```

Monitor: https://github.com/neokree/zeroclaw/actions/workflows/release-nexus.yml

- [ ] **Step 2: After CI succeeds, deploy to Bison Pro**

Follow deploy procedure (SSH + update-runner or manual swap).

After deploy, verify in ZeroClaw logs that pipeline processes a turn without errors:
```bash
ssh bison-pro "pm2 logs zeroclaw-gateway --lines 50 --nostream 2>&1 | grep pipeline"
```

Expected: log lines like `Saved N foresights for episode ...` — no errors about embedding.
