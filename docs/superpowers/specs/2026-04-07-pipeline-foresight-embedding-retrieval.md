# Pipeline Foresight — Embedding Similarity Retrieval

**Date**: 2026-04-07
**Author**: Fabio Biola + Claude
**Status**: Draft
**Branch**: `nexus`

## Summary

Complete the memory pipeline by adding embedding-based similarity retrieval for `<foresight>` entries, matching EverMemOS behavior. Currently foresights are retrieved by `ORDER BY created_at DESC`; this change retrieves them by cosine similarity to the user's current query, with graceful fallback to recency when the embedder is unavailable.

Episodes remain recency-based — this is intentional and matches EverMemOS.

---

## Architecture

### Files Modified

| File | Change |
|------|--------|
| `src/memory/pipeline/mod.rs` | Add `embedder: Option<Arc<dyn EmbeddingProvider>>` to `Pipeline` struct and `new()` |
| `src/memory/pipeline/store.rs` | `save_foresights` writes embedding BLOB; `get_active_foresights` accepts `query_embedding: Option<Vec<f32>>` and ranks in-memory |
| `src/memory/pipeline/formatter.rs` | Receives `query_embedding: Option<Vec<f32>>` from Pipeline, passes to store |
| `src/channels/mod.rs` | Re-creates `EmbeddingProvider` from `config.memory` fields and passes it to `Pipeline::new()` |

**No new files.** The `EmbeddingProvider` trait (`src/memory/embeddings.rs`) and llama-server are reused without changes.

### Key Design Decisions

**EmbeddingProvider injection**: The `EmbeddingProvider` inside `SqliteMemory` is a private field with no public accessor. At the `Pipeline::new()` call site in `channels/mod.rs`, the only handle is `Arc<dyn Memory>`. Therefore, `channels/mod.rs` re-creates the `EmbeddingProvider` from the same `config.memory` fields (`embedding_provider`, `embedding_model`, `embedding_dimensions`) already used by `build_sqlite_memory`. This is consistent with how `SkillCreator` receives its embedding provider at the same call site. The result is wrapped in `Option` — `None` if `embedding_provider` is not configured.

**Query embedding in build_context**: `Pipeline::build_context` generates the query embedding before calling the formatter, then passes `Option<Vec<f32>>` as a new 5th parameter to `formatter.build_xml_context`. The formatter remains passive. There is one call site for `build_xml_context` (in `Pipeline::build_context`), so the signature change is contained.

**BLOB serialization**: `Vec<f32>` as raw little-endian bytes via `vector::vec_to_bytes` / `vector::bytes_to_vec`, consistent with `src/memory/sqlite.rs`.

---

## Data Flow

### At Save Time (`process_turn_inner`)

After foresight extraction:

```
if embedder.is_some() && foresights non-empty:
  contents = [f.content for f in foresights]
  embeddings = embedder.embed(&contents).await  // single batch HTTP call
  → serialize each Vec<f32> as raw bytes via vector::vec_to_bytes
else:
  embeddings = [None; foresights.len()]

save_foresights(rows with embedding BLOB or NULL)
```

A single batch call to `embed(&[...])` produces one HTTP request to llama-server regardless of foresight count. If the batch call fails entirely, all foresights are saved with `NULL` BLOB — non-blocking. The `save_foresights` SQL INSERT expands from 8 to 9 columns to include `embedding`.

### At Retrieval Time (`Pipeline::build_context`)

```
Pipeline::build_context(user_query, ...):
  query_embedding = embedder.embed_one(user_query).await.ok()  // Ok→Some, Err→None
  formatter.build_xml_context(..., query_embedding)

store.get_active_foresights(user_id, limit, query_embedding):
  if query_embedding.is_some():
    SELECT id, episode_id, user_id, content, evidence,
           start_time, end_time, created_at, embedding   ← 9 columns (was 8)
    FROM foresights WHERE active + user filter
    // no ORDER BY — ranking done in Rust
    for each with BLOB: deserialize via vector::bytes_to_vec,
                        compute cosine_similarity(query_emb, foresight_emb)
    for each without BLOB: score = -1.0
    sort by score DESC, take top-N
  else:
    ORDER BY created_at DESC LIMIT N  (current behavior)
```

Cosine similarity is a pure in-memory function — O(n) over active foresights. With daily use, a few hundred active foresights at most; negligible cost.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| llama-server unavailable at save time | Foresight saved with `embedding = NULL`, log warning, no error propagated |
| llama-server unavailable at retrieval | `query_embedding = None` → fallback to `ORDER BY created_at DESC` |
| Batch embedding failure at save time | All foresights for that flush saved with `NULL` BLOB — does not block flush, log warning |
| Corrupted BLOB or wrong dimension | `cosine_similarity` returns `None` → foresight treated as score -1.0 |
| `EmbeddingProvider` not configured in `Pipeline::new` | `embedder = None` → entire system behaves as today, zero impact |

No new critical failure points — every error degrades gracefully to current behavior.

---

## Relationship with Existing Systems

| System | Impact |
|--------|--------|
| Episode retrieval (`recent_episodes`) | Unchanged — recency-based, correct per EverMemOS |
| `<facts>` via `Memory::recall()` | Unchanged — still uses existing hybrid BM25+cosine |
| Profile retrieval | Unchanged — by `user_id` direct lookup |
| llama-server (EmbeddingGemma-300M) | Reused — same provider, no config changes needed |
| `ForesightRow` struct | Adds `embedding: Option<Vec<u8>>` field; `save_foresights` SQL expands from 8 to 9 columns; `get_active_foresights` SELECT adds `embedding` as 9th column |
| `src/memory/vector.rs` | Reused (no changes) — `vec_to_bytes`, `bytes_to_vec`, `cosine_similarity` already public |
