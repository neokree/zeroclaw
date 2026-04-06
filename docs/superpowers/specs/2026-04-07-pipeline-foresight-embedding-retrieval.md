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
| `src/channels/mod.rs` | Passes existing `EmbeddingProvider` instance to `Pipeline::new()` |

**No new files.** The `EmbeddingProvider` trait (`src/memory/embeddings.rs`) and the existing llama-server instance are reused without changes.

### Key Design Decision

Embedding of the query is generated in `Pipeline::build_context` (which owns the embedder), then passed as `Option<Vec<f32>>` through `formatter.build_xml_context` → `store.get_active_foresights`. The formatter remains passive — all embedding logic lives in the Pipeline layer.

BLOB serialization: `Vec<f32>` as raw little-endian bytes, consistent with `src/memory/sqlite.rs`.

---

## Data Flow

### At Save Time (`process_turn_inner`)

After foresight extraction:

```
for each extracted foresight:
  if embedder.is_some():
    embedding = embedder.embed(foresight.content).await
    → serialize Vec<f32> as raw bytes
  else:
    embedding = None

save_foresights(rows with embedding BLOB or NULL)
```

All embeddings are generated in parallel via `join_all` — one batch of HTTP calls to llama-server. A failure on a single foresight saves that row with `NULL` BLOB without blocking the flush.

### At Retrieval Time (`Pipeline::build_context`)

```
Pipeline::build_context(user_query, ...):
  query_embedding = embedder.embed(user_query).await  // Option<Vec<f32>>
  formatter.build_xml_context(..., query_embedding)

store.get_active_foresights(user_id, limit, query_embedding):
  if query_embedding.is_some():
    load ALL active foresights with BLOBs
    for each with BLOB: compute cosine_similarity(query_emb, foresight_emb)
    for each without BLOB: score = -1.0
    sort by score DESC, take top-N
  else:
    fallback: ORDER BY created_at DESC
```

Cosine similarity is a pure in-memory function — O(n) over active foresights. With daily use, a few hundred active foresights at most; negligible cost.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| llama-server unavailable at save time | Foresight saved with `embedding = NULL`, log warning, no error propagated |
| llama-server unavailable at retrieval | `query_embedding = None` → fallback to `ORDER BY created_at DESC` |
| Partial embedding failure (1 of N foresights) | Others saved with BLOB, failed one saved with NULL — does not block flush |
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
| `ForesightRow` struct | Adds optional `embedding: Option<Vec<u8>>` field |
