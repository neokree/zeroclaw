# ZeroClaw Memory Pipeline — Design Spec

**Date**: 2026-04-05
**Author**: Fabio Biola + Claude
**Status**: Draft
**Branch**: `nexus`
**Inspired by**: [EverMemOS](https://github.com/EverMind-AI/EverMemOS) (Apache 2.0)

## Summary

Add a **Memory Pipeline** to ZeroClaw — a modular per-turn system inspired by EverMemOS that introduces boundary detection, multi-type memory extraction, and XML-formatted memory injection.

This **coexists** with the existing nightly consolidation cron (`src/cron/consolidation.rs`) and the existing `auto_save` mechanism. It does not replace them — it adds a smarter real-time extraction layer on top.

### Goals
- Smarter memory extraction via boundary detection (flush when a conversation segment ends, not every turn)
- Multi-type extraction: Episode (narrative), EventLog (atomic facts), Foresight (predictions), Profile (user traits)
- Structured memory injection in XML format (replacing flat `[Memory context]` block)
- Multi-language prompt support (Italian as primary)
- Upstream-friendly: designed to be contributed back to zeroclaw-labs

### Phased Rollout

| Phase | Features |
|-------|----------|
| **1** | Boundary Detection, Buffer, Episode + EventLog extraction, XML Injection, Multi-language prompts |
| **2** | Foresight extraction (behavioral predictions with time bounds) |
| **3** | Profile extraction (structured user attributes with incremental merging) |

---

## Architecture

### Pipeline Struct

The `Pipeline` struct holds all dependencies needed for autonomous operation:

```rust
pub struct Pipeline {
    provider: Arc<dyn Provider>,       // for LLM calls (boundary, extraction)
    embedding: Arc<dyn EmbeddingProvider>, // for generating embeddings
    memory: Arc<dyn Memory>,           // for Memory::recall() in formatter
    store: PipelineStore,              // manages pipeline-specific tables
    config: PipelineConfig,            // from [memory.pipeline] config
}
```

Constructed in `src/channels/mod.rs` during channel initialization, using existing `Provider` and `Memory` instances. The pipeline receives the same `Provider` used by the agent — if `extraction_model` is configured, it passes that model name to `Provider::chat()` instead of the default.

### User Identity

`user_id` throughout this spec refers to the **channel's sender ID** (e.g., Telegram user ID `"18189716"`, Discord user ID, etc.). This is the same identifier available in `channels/mod.rs` when processing incoming messages.

For single-user deployments (like Nexus), there is only one `user_id`. For multi-channel setups where the same person uses multiple channels, user identity mapping is out of scope for Phase 1 — each channel ID is treated as a separate user.

### Module Structure

New self-contained module at `src/memory/pipeline/`:

```
src/memory/pipeline/
├── mod.rs              — Pipeline struct, orchestration
├── buffer.rs           — Buffer: accumulates messages in memcell_buffer table
├── boundary.rs         — BoundaryDetector: LLM + hard limits
├── extractor/
│   ├── mod.rs          — Extractor trait + orchestration
│   ├── episode.rs      — EpisodeExtractor: narrative generation
│   ├── event_log.rs    — EventLogExtractor: atomic facts
│   ├── foresight.rs    — ForesightExtractor: predictions (Phase 2)
│   └── profile.rs      — ProfileExtractor: user traits (Phase 3)
├── store.rs            — PipelineStore: manages pipeline-specific tables
├── formatter.rs        — Formatter: builds XML injection block
├── merger.rs           — ProfileMerger: incremental profile updates (Phase 3)
└── config.rs           — PipelineConfig: [memory.pipeline] config section
```

Localized prompts embedded via `include_str!()` — compiled into the binary, organized as:
```
src/memory/pipeline/prompts/
├── en/
│   ├── boundary.txt
│   ├── episode.txt
│   ├── event_log.txt
│   ├── foresight.txt      (Phase 2)
│   └── profile.txt        (Phase 3)
└── it/
    ├── boundary.txt
    ├── episode.txt
    ├── event_log.txt
    ├── foresight.txt      (Phase 2)
    └── profile.txt        (Phase 3)
```

### Files Modified (Existing)

| File | Change |
|------|--------|
| `src/agent/loop_/context.rs` | Replace `[Memory context]` block with XML injection from `formatter.rs` |
| `src/channels/mod.rs` | Construct `Pipeline`, call `append_to_buffer` + `process_turn` |
| `src/memory/sqlite.rs` | Add pipeline table creation in schema init |

**Not modified**: `src/memory/traits.rs` — the pipeline uses its own `PipelineStore`, separate from the `Memory` trait. Existing categories (`Core`, `Daily`, `Conversation`, `Custom`) remain untouched.

---

## Phase 1: Boundary Detection + Episode + EventLog

### Database Schema (Phase 1)

#### `memcell_buffer` — Temporary conversation buffer

```sql
CREATE TABLE memcell_buffer (
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL,
  role        TEXT NOT NULL,          -- 'user' | 'assistant'
  content     TEXT NOT NULL,
  char_count  INTEGER NOT NULL,      -- length of content, for token estimation
  created_at  TEXT NOT NULL
);
CREATE INDEX idx_memcell_buffer_session ON memcell_buffer(session_id, created_at);
```

#### `episodes` — Narrative memories

```sql
CREATE TABLE episodes (
  id          TEXT PRIMARY KEY,
  session_id  TEXT,
  user_id     TEXT,                   -- channel sender_id
  subject     TEXT,                   -- title (10-20 words)
  summary     TEXT,                   -- brief abstract
  episode     TEXT NOT NULL,          -- full 3rd-person narrative
  embedding   BLOB,
  created_at  TEXT NOT NULL
);
```

#### `event_logs` — Atomic facts

```sql
CREATE TABLE event_logs (
  id          TEXT PRIMARY KEY,
  episode_id  TEXT REFERENCES episodes(id),
  atomic_fact TEXT NOT NULL,          -- single atomic fact sentence
  time_ref    TEXT,                   -- time reference from conversation
  embedding   BLOB,
  created_at  TEXT NOT NULL
);
CREATE INDEX idx_event_logs_episode ON event_logs(episode_id);
```

**Design note**: Each episode produces multiple `event_logs` rows (one per atomic fact), linked via `episode_id`. This mirrors EverMemOS where EventLog has `parent_id` pointing to MemCell/Episode.

### Migration (Existing Data)

Existing `memories` table is **kept as-is**. No renaming, no destructive changes. The `episodes` table starts **empty** — existing `core` memories are key-value pairs (not narratives) and would not produce meaningful episodes. New episodes will be generated from real conversations going forward.

1. Backup: `cp brain.db brain_backup_YYYYMMDD.db`
2. Create new tables (`memcell_buffer`, `episodes`, `event_logs`)

### Boundary Detection

Inspired by EverMemOS `ConvMemCellExtractor`. Two layers:

#### Layer 1: Hard Limits (force flush, no LLM call)

Token estimation uses `chars / 4` heuristic (no tiktoken in Rust; approximation is sufficient for a threshold check). The `char_count` column in `memcell_buffer` stores pre-computed content length. Total estimated tokens = `SUM(char_count) / 4` for the session.

```
IF estimated_tokens >= hard_token_limit (default 4096):
    → force flush, keep most recent messages for new buffer
IF message_count >= hard_message_limit (default 30):
    → force flush
```

#### Layer 2: LLM Semantic Boundary Detection

Only runs if hard limits are not hit.

**Prompt** (adapted from EverMemOS `CONV_BOUNDARY_DETECTION_PROMPT`):

```
Analyze this conversation and determine if the NEW MESSAGES represent
a topic boundary (end of a coherent conversation segment).

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
{
  "reasoning": "one sentence",
  "should_end": true/false,
  "confidence": 0.0-1.0,
  "topic_summary": "if should_end, summarize; else empty string"
}
```

**Smart mask** (from EverMemOS): when buffer has >5 messages, exclude the last message from history when evaluating boundary. This prevents the LLM from being biased by the most recent exchange.

**LLM call**: extraction_model, temperature 0.1, ~100 output tokens.

### Multi-Phase Extraction (Phase 1)

When boundary triggers, two **parallel** LLM calls run on the buffer content:

#### 1. Episode Extraction

Adapted from EverMemOS `EPISODE_GENERATION_PROMPT`:

```
Convert this conversation into a concise third-person narrative.

Requirements:
- Chronological factual record with all important info
- Preserve: person names, dates, locations, emotions, decisions, outcomes
- Time references as: "relative time (absolute date)"
- Record frequency information (patterns, repetitions)
- Remove redundancies while keeping all facts
- Do NOT include greetings/small talk

Respond ONLY with valid JSON:
{
  "subject": "concise title (10-20 words)",
  "episode": "third-person narrative",
  "summary": "2-3 sentence abstract"
}
```

#### 2. EventLog Extraction

Adapted from EverMemOS `EVENT_LOG_PROMPT`:

```
Extract atomic facts from this conversation.

Rules:
- Each fact = exactly one coherent unit (action, emotion, plan, decision)
- Single complete sentence, third-person form
- Explicit attribution: always state WHO said/did what
- Resolve pronouns to specific names
- Preserve timestamps; resolve relative times: "yesterday (2026-04-04)"
- Filter out: greetings, "okay", low-value chatter

Respond ONLY with valid JSON:
{
  "atomic_facts": [
    {"fact": "sentence", "time": "time reference or null"},
    ...
  ]
}
```

### Data Flow (Phase 1)

```
[User message arrives in channels/mod.rs]
      |
      |-- Pipeline::append_to_buffer(user_msg, session_id)
      |     → INSERT into memcell_buffer (role='user')
      |
      |-- formatter::build_xml_context(query, user_id)
      |     |-- <episodic>: last N episodes (by date)
      |     |-- <facts>: Memory::recall(query) from memories table
      |     \-- prepend XML block to user message
      |
      v
[LLM call] -> response to user
      |
      |-- Pipeline::append_to_buffer(assistant_msg, session_id)    [SYNC]
      |     → INSERT into memcell_buffer (role='assistant')
      |
      v (tokio::spawn — fire-and-forget)
Pipeline::process_turn(session_id)
  |
  |-- Check hard limits (estimated tokens, message count)
  |     → if exceeded: force flush
  |
  |-- [IF no force flush] BoundaryDetector::detect(buffer)
  |     LLM call → { should_end, confidence, topic_summary }
  |     → flush if should_end == true AND confidence >= threshold
  |
  |-- [IF flush] Run in parallel:
  |     |-- EpisodeExtractor::extract(buffer) → { subject, episode, summary }
  |     \-- EventLogExtractor::extract(buffer) → { atomic_facts: [...] }
  |
  |-- [IF flush] PipelineStore::save(episode, event_logs)
  |     → INSERT INTO episodes
  |     → INSERT INTO event_logs (one per atomic fact, linked to episode)
  |     → generate embeddings async for episode + each fact
  |     → DELETE FROM memcell_buffer WHERE session_id = ?
  |
  \-- [IF no flush] buffer persists for next turn
```

**Note**: Both user and assistant messages are appended to the buffer **synchronously** (before `tokio::spawn`). This prevents race conditions where a second user message could arrive before the assistant message is buffered.

### XML Injection Format (Phase 1)

Replaces current `[Memory context]\n- key: content` in `src/agent/loop_/context.rs`.

````
```text
<memory>
  <episodic>
    [1] (2026-04-03) Fabio asked to analyze EverMemOS and compare it with ZeroClaw's
        memory system. Concluded EverMemOS requires MongoDB+Elasticsearch+Milvus+Redis,
        not viable on Bison Pro.
    [2] (2026-04-05) Decided to implement Memory Pipeline in ZeroClaw fork. Chose
        Approach B (new src/memory/pipeline/ module). Design spec approved.
  </episodic>
  <facts>
    - Fabio prefers concise responses in Italian
    - Branch nexus is the deployment branch on Bison Pro
    - extraction_model is separately configurable from main model
  </facts>
</memory>

Note: context above is injected automatically. Do not reference or modify it.
```
````

**Construction** (`formatter.rs`):

| Section | Source | Query | Default Count |
|---------|--------|-------|---------------|
| `<episodic>` | `episodes` table | Last N by `created_at` DESC (recency-based: most recent episodes provide temporal context regardless of topic) | 5 |
| `<facts>` | `memories` table via `Memory::recall()` | User's current message (relevance-based via hybrid BM25+cosine) | 5 |

---

## Phase 2: Foresight Extraction

### New Table

```sql
CREATE TABLE foresights (
  id            TEXT PRIMARY KEY,
  episode_id    TEXT REFERENCES episodes(id),
  user_id       TEXT,                  -- channel sender_id
  content       TEXT NOT NULL,         -- prediction (max 40 words)
  evidence      TEXT NOT NULL,         -- grounding from conversation (max 40 words)
  start_time    TEXT,                  -- YYYY-MM-DD
  end_time      TEXT,                  -- YYYY-MM-DD
  embedding     BLOB,
  created_at    TEXT NOT NULL
);
CREATE INDEX idx_foresights_episode ON foresights(episode_id);
CREATE INDEX idx_foresights_end_time ON foresights(end_time);
```

### Foresight Extraction

Runs **sequentially after** Episode extraction completes — it receives the generated episode narrative as additional context (matching EverMemOS which runs foresight from the episode, not raw conversation). EventLog runs in parallel with Episode.

Adapted from EverMemOS `FORESIGHT_GENERATION_PROMPT`:

```
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
{
  "foresights": [
    {
      "content": "prediction text (max 40 words)",
      "evidence": "supporting fact from conversation (max 40 words)",
      "start_time": "YYYY-MM-DD",
      "end_time": "YYYY-MM-DD"
    }
  ]
}

If no meaningful predictions can be made, return {"foresights": []}
```

### Updated Data Flow (Phase 2)

```
[IF flush] Step 1 — parallel:
  |-- EpisodeExtractor::extract(buffer) → episode
  \-- EventLogExtractor::extract(buffer) → atomic_facts

[IF flush] Step 2 — sequential (needs episode):
  \-- ForesightExtractor::extract(buffer, episode) → foresights

[IF flush] Step 3 — save all:
  → INSERT INTO episodes
  → INSERT INTO event_logs
  → INSERT INTO foresights (one per prediction)
  → generate embeddings
  → clear buffer
```

### Updated XML Injection (Phase 2)

````
```text
<memory>
  <episodic>
    [1] (2026-04-03) ...
    [2] (2026-04-05) ...
  </episodic>
  <facts>
    - ...
  </facts>
  <foresight>
    - User will likely want to start Rust implementation of memory pipeline this week
    - User may need to test boundary detection on real Telegram conversations
  </foresight>
</memory>
```
````

**Foresight filtering**: only show foresights where `end_time >= today` (not expired). Ordered by relevance to current query via embedding similarity.

| Section | Source | Query | Default Count |
|---------|--------|-------|---------------|
| `<foresight>` | `foresights` table, filtered `end_time >= today` | Embedding similarity to user message | 3 |

---

## Phase 3: Profile Extraction

### New Table

```sql
CREATE TABLE profiles (
  id              TEXT PRIMARY KEY,
  user_id         TEXT NOT NULL UNIQUE,  -- channel sender_id
  profile_data    TEXT NOT NULL,          -- JSON: structured profile attributes
  confidence      REAL DEFAULT 0.0,
  version         INTEGER DEFAULT 1,
  episode_count   INTEGER DEFAULT 0,     -- total episodes processed for this profile
  last_episode_id TEXT,                  -- latest episode that contributed
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_profiles_user ON profiles(user_id);
```

### Profile Attributes (from EverMemOS)

The `profile_data` JSON contains these structured attributes:

```json
{
  "user_name": "string",
  "hard_skills": [{"value": "Rust", "level": "advanced", "evidences": ["ep_123"]}],
  "soft_skills": [{"value": "Leadership", "level": "intermediate", "evidences": ["ep_456"]}],
  "personality": [{"value": "cautious, prefers understanding before acting", "evidences": ["ep_789"]}],
  "way_of_decision_making": [{"value": "asks for options before choosing", "evidences": []}],
  "interests": [{"value": "AI agents", "evidences": []}],
  "user_goal": [{"value": "build Nexus AI assistant on Bison Pro", "evidences": []}],
  "work_responsibility": [{"value": "full-stack AI infrastructure", "evidences": []}],
  "working_habit_preference": [{"value": "brainstorm before implementing", "evidences": []}],
  "tendency": [{"value": "prefers Italian communication", "evidences": []}],
  "motivation_system": [{"value": "solving hard technical problems", "level": "advanced", "evidences": []}],
  "fear_system": [{"value": "bricking devices", "level": "advanced", "evidences": []}],
  "value_system": [{"value": "understanding over speed", "level": "advanced", "evidences": []}],
  "humor_use": [{"value": "occasional, technical", "level": "intermediate", "evidences": []}],
  "colloquialism": [{"value": "informal Italian", "level": "advanced", "evidences": []}]
}
```

Attributes with `level` field: `hard_skills`, `soft_skills`, `motivation_system`, `fear_system`, `value_system`, `humor_use`, `colloquialism`.

Attributes without `level`: `personality`, `way_of_decision_making`, `interests`, `user_goal`, `work_responsibility`, `working_habit_preference`, `tendency`.

### Profile Extraction

Runs **sequentially after** Episode extraction. Does NOT run every flush — runs periodically to avoid excessive LLM calls.

**Trigger algorithm**:
1. After saving an episode, increment `profiles.episode_count` for the relevant `user_id` (UPSERT: create profile row if first episode for this user).
2. If `episode_count % profile_update_interval == 0` (default interval: 5), trigger profile extraction.
3. Gather the last `profile_update_interval` episodes for this `user_id` as input.
4. Load current `profile_data` JSON (empty `{}` if first extraction).
5. Call LLM with current profile + recent episodes → get updated profile.
6. Merge via `ProfileMerger`, increment `version`, update `updated_at`.

Adapted from EverMemOS `CONVERSATION_PROFILE_EXTRACTION_PROMPT`:

```
Analyze these conversation episodes and extract/update the user profile.

Current profile (may be empty if first extraction):
{current_profile_json}

New episodes to analyze:
{recent_episodes}

For each attribute, provide evidence (episode IDs that support it).
For skill-like attributes, assign a level: "beginner", "intermediate", "advanced", "expert".

Respond ONLY with valid JSON matching this schema:
{
  "user_name": "...",
  "hard_skills": [{"value": "...", "level": "...", "evidences": ["ep_id"]}],
  "soft_skills": [...],
  "personality": [{"value": "...", "evidences": ["ep_id"]}],
  ...
}

Rules:
- Merge with existing profile: update levels if evidence is stronger, add new attributes
- Keep highest level when conflicting evidence exists
- Truncate evidences to max 10 per attribute
- Only add attributes with clear evidence, never speculate
```

### Profile Merger (`merger.rs`)

Incremental merge strategy (from EverMemOS `ProfileMemoryMerger`):

1. **Leveled attributes** (skills, motivation, etc.): keep highest level across merges
2. **Non-leveled attributes** (personality, goals, etc.): merge lists, concatenate evidences
3. **Evidence truncation**: max 10 evidence items per attribute, prioritize recent
4. **Version bump**: increment `version` on each merge, update `updated_at`

### Updated Data Flow (Phase 3 additions)

```
[IF flush] Step 1 — parallel:
  |-- EpisodeExtractor::extract(buffer) → episode
  \-- EventLogExtractor::extract(buffer) → atomic_facts

[IF flush] Step 2 — sequential (needs episode):
  |-- ForesightExtractor::extract(buffer, episode) → foresights
  \-- Increment profiles.episode_count for user_id

[IF flush] Step 3 — conditional:
  \-- [IF episode_count % profile_update_interval == 0]
        ProfileExtractor::extract(recent_episodes, current_profile)
            → UPSERT INTO profiles

[IF flush] Step 4 — save all:
  → INSERT INTO episodes, event_logs, foresights
  → UPSERT profiles (if triggered)
  → generate embeddings
  → clear buffer
```

### Updated XML Injection (Phase 3)

````
```text
<memory>
  <trait>
    Name: Fabio Biola
    Skills: Rust (advanced), Python (intermediate), Android modding (advanced)
    Personality: cautious, prefers understanding before acting
    Goals: build Nexus AI assistant on Bison Pro
    Preferences: Italian communication, brainstorm before implementing
    Interests: AI agents, hardware hacking
  </trait>
  <episodic>
    [1] (2026-04-03) ...
    [2] (2026-04-05) ...
  </episodic>
  <facts>
    - ...
  </facts>
  <foresight>
    - ...
  </foresight>
</memory>

Note: context above is injected automatically. Do not reference or modify it.
```
````

**Profile injection**: `<trait>` is rendered as a human-readable summary from `profile_data` JSON. One profile per user. Placed first in `<memory>` block (like EverMemOS puts Profile first).

| Section | Source | Query | Default Count |
|---------|--------|-------|---------------|
| `<trait>` | `profiles` table | By `user_id` (channel sender_id from message context) | 1 |
| `<episodic>` | `episodes` table | Last N by date | 5 |
| `<facts>` | `memories` table via `Memory::recall()` | User's current message | 5 |
| `<foresight>` | `foresights` table, filtered `end_time >= today` | Embedding similarity | 3 |

---

## Error Handling (All Phases)

| Failure | Behavior |
|---------|----------|
| Boundary detection LLM fails/times out | Skip flush, buffer persists. Log warning. |
| Extraction LLM returns malformed JSON | Log error, do NOT flush buffer. Retry next turn. |
| Buffer exceeds `hard_token_limit` | **Force flush**: skip boundary LLM, go directly to extraction. Truncate buffer from start (keep most recent messages). |
| `extraction_model` not found | Fall back to main model. Log warning at startup. |
| Episode/EventLog INSERT fails | Log error, do NOT clear buffer. Preserved for retry. |
| Foresight extraction fails (Phase 2) | Log warning, save episode+eventlogs anyway. Non-blocking. |
| Profile extraction fails (Phase 3) | Log warning, save episode+eventlogs+foresights anyway. Non-blocking. |
| No meaningful content in buffer | Boundary detection returns `should_end: false` → buffer accumulates. |
| Abandoned session (no messages for >24h) | Nightly cleanup: force flush any buffer with `MAX(created_at) < NOW - 24h`, then extract. If extraction also fails, delete stale buffer rows and log warning. |

---

## Configuration

New section in `~/.zeroclaw/config.toml`:

```toml
[memory.pipeline]
enabled = true

# Model for all pipeline LLM calls (boundary, extraction).
# Resolved through the same [provider]. If omitted, uses main model.
extraction_model = "moonshotai/kimi-k2.5"

# Language for internal prompts
prompt_language = "it"              # "en" | "it"

# --- Boundary Detection ---
boundary_threshold = 0.6           # confidence threshold (0.0-1.0)
hard_token_limit = 4096            # force flush if estimated tokens exceed
hard_message_limit = 30            # force flush if message count exceeds

# --- Buffer Cleanup ---
buffer_ttl_hours = 24              # force flush abandoned buffers after this

# --- Injection ---
max_episodic_entries = 5
max_facts_entries = 5
max_foresight_entries = 3           # Phase 2

# --- Profile (Phase 3) ---
profile_update_interval = 5         # update profile every N episodes
```

### Defaults

If `[memory.pipeline]` section is absent or `enabled = false`: pipeline disabled, current behavior preserved. **Zero breaking changes.** Runtime toggle only — no compile-time feature flags.

### Model Routing

`extraction_model` is resolved through the **same provider** configured in `[provider]`. The pipeline calls `Provider::chat()` with the `extraction_model` name instead of the default model. If the model string is invalid or the provider returns an error, the pipeline falls back to the main model and logs a warning.

---

## Relationship with Existing Systems

| System | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Nightly consolidation cron** | Unaffected. Can optionally trigger buffer cleanup for abandoned sessions. | Unaffected | Unaffected |
| **auto_save** (raw messages → `conversation`) | Unaffected | Unaffected | Unaffected |
| **Memory::recall()** (hybrid BM25+cosine) | Still used for `<facts>` in XML injection | Same | Same |
| **MEMORY.md workspace injection** | Unaffected | Unaffected | Unaffected |
| **`[Memory context]` block** | Replaced by XML `<memory>` | Same | Same |

---

## Upstream Contribution Plan

Designed to be contributed back to [zeroclaw-labs/zeroclaw](https://github.com/zeroclaw-labs/zeroclaw):

- Self-contained `src/memory/pipeline/` module, no changes to `Memory` trait
- Opt-in via config (disabled by default, runtime toggle)
- Clean separation from existing backends and nightly consolidation
- Phased: each phase is independently useful and mergeable
