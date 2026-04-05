# Memory Pipeline Phase 3 — Profile Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Profile extraction (structured user attributes with incremental merging) to the Memory Pipeline, with XML injection of user traits.

**Architecture:** Extends `src/memory/pipeline/` (built in Phase 1+2) with a `ProfileExtractor` that runs conditionally every N episodes, a `ProfileMerger` for incremental updates (keeping highest skill levels, merging evidence lists), and a `profiles` table. The `Formatter` is updated to include a `<trait>` section as the first element in the XML block. Profile extraction is non-blocking — if it fails, episode + event_log + foresight are still saved.

**Tech Stack:** Rust, SQLite (rusqlite), tokio, serde_json, chrono

**Spec:** `docs/superpowers/specs/2026-04-05-zeroclaw-memory-pipeline-design.md` (Phase 3 section)

**Prerequisite:** Phase 1 + Phase 2 must be implemented.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/memory/pipeline/extractor/profile.rs` | `ProfileExtractor`: LLM call → `ProfileData` |
| `src/memory/pipeline/merger.rs` | `ProfileMerger`: incremental merge of old + new profile |
| `src/memory/pipeline/profile_types.rs` | `ProfileData`, `LeveledAttribute`, `Attribute` serde structs |
| `src/memory/pipeline/prompts/en/profile.txt` | English profile extraction prompt |
| `src/memory/pipeline/prompts/it/profile.txt` | Italian profile extraction prompt |

### Modified Files

| File | Change |
|------|--------|
| `src/memory/pipeline/store.rs` | Add `profiles` table creation + `get_profile()` + `upsert_profile()` + `increment_episode_count()` + `get_recent_episodes()` |
| `src/memory/pipeline/config.rs` | Add `profile_update_interval: usize` field (default 5) |
| `src/memory/pipeline/extractor/mod.rs` | Add `pub mod profile;`, update orchestration with conditional profile trigger |
| `src/memory/pipeline/formatter.rs` | Add `<trait>` section as first element in XML block |
| `src/memory/pipeline/mod.rs` | Update `process_turn()` with profile trigger algorithm |

---

## Chunk 1: Profile Types + Merger

### Task 1: Define profile data types

**Files:**
- Create: `src/memory/pipeline/profile_types.rs`
- Modify: `src/memory/pipeline/mod.rs` (add `pub mod profile_types;`)

- [ ] **Step 1: Define attribute structs**

```rust
// src/memory/pipeline/profile_types.rs
use serde::{Deserialize, Serialize};

/// Attribute with skill level (hard_skills, soft_skills, motivation_system, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeveledAttribute {
    pub value: String,
    pub level: String,    // "beginner" | "intermediate" | "advanced" | "expert"
    #[serde(default)]
    pub evidences: Vec<String>,
}

/// Attribute without level (personality, interests, goals, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Attribute {
    pub value: String,
    #[serde(default)]
    pub evidences: Vec<String>,
}

/// Full profile data structure — serialized as JSON in profiles.profile_data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProfileData {
    #[serde(default)]
    pub user_name: Option<String>,

    // Leveled attributes
    #[serde(default)]
    pub hard_skills: Vec<LeveledAttribute>,
    #[serde(default)]
    pub soft_skills: Vec<LeveledAttribute>,
    #[serde(default)]
    pub motivation_system: Vec<LeveledAttribute>,
    #[serde(default)]
    pub fear_system: Vec<LeveledAttribute>,
    #[serde(default)]
    pub value_system: Vec<LeveledAttribute>,
    #[serde(default)]
    pub humor_use: Vec<LeveledAttribute>,
    #[serde(default)]
    pub colloquialism: Vec<LeveledAttribute>,

    // Non-leveled attributes
    #[serde(default)]
    pub personality: Vec<Attribute>,
    #[serde(default)]
    pub way_of_decision_making: Vec<Attribute>,
    #[serde(default)]
    pub interests: Vec<Attribute>,
    #[serde(default)]
    pub user_goal: Vec<Attribute>,
    #[serde(default)]
    pub work_responsibility: Vec<Attribute>,
    #[serde(default)]
    pub working_habit_preference: Vec<Attribute>,
    #[serde(default)]
    pub tendency: Vec<Attribute>,
}
```

- [ ] **Step 2: Add level ordering helper**

```rust
/// Level ordering for comparison: higher index = higher skill.
/// All leveled attributes use this same vocabulary (beginner → expert).
/// The prompt explicitly instructs the LLM to use only these four levels.
const LEVEL_ORDER: &[&str] = &["beginner", "intermediate", "advanced", "expert"];

pub fn level_rank(level: &str) -> usize {
    // Normalize: map legacy/alternative levels to standard vocabulary
    let normalized = match level.to_lowercase().as_str() {
        "low" | "beginner" => "beginner",
        "moderate" | "intermediate" => "intermediate",
        "high" | "advanced" => "advanced",
        "very_high" | "expert" => "expert",
        _ => "beginner",
    };
    LEVEL_ORDER.iter().position(|&l| l == normalized).unwrap_or(0)
}
```

- [ ] **Step 3: Add `pub mod profile_types;` to `mod.rs`**

- [ ] **Step 4: Compile and verify**

Run: `cargo check`

- [ ] **Step 5: Commit**

```bash
git add src/memory/pipeline/profile_types.rs src/memory/pipeline/mod.rs
git commit -m "feat(pipeline): add ProfileData types with leveled and non-leveled attributes"
```

---

### Task 2: Implement ProfileMerger

**Files:**
- Create: `src/memory/pipeline/merger.rs`
- Modify: `src/memory/pipeline/mod.rs` (add `pub mod merger;`)

- [ ] **Step 1: Implement leveled attribute merge**

Merges two lists of `LeveledAttribute`. When the same `value` exists in both, keeps the **highest level**. Concatenates evidences and truncates to max 10.

```rust
// src/memory/pipeline/merger.rs
use super::profile_types::*;

const MAX_EVIDENCES: usize = 10;

fn merge_leveled(existing: &[LeveledAttribute], incoming: &[LeveledAttribute]) -> Vec<LeveledAttribute> {
    let mut merged: Vec<LeveledAttribute> = existing.to_vec();

    for new_attr in incoming {
        if let Some(existing_attr) = merged.iter_mut().find(|a| a.value == new_attr.value) {
            // Keep highest level
            if level_rank(&new_attr.level) > level_rank(&existing_attr.level) {
                existing_attr.level = new_attr.level.clone();
            }
            // Merge evidences, truncate to MAX_EVIDENCES (keep most recent = last)
            existing_attr.evidences.extend(new_attr.evidences.clone());
            let len = existing_attr.evidences.len();
            if len > MAX_EVIDENCES {
                existing_attr.evidences = existing_attr.evidences[len - MAX_EVIDENCES..].to_vec();
            }
        } else {
            merged.push(new_attr.clone());
        }
    }

    merged
}
```

- [ ] **Step 2: Implement non-leveled attribute merge**

Same logic but no level comparison.

```rust
fn merge_attributes(existing: &[Attribute], incoming: &[Attribute]) -> Vec<Attribute> {
    let mut merged: Vec<Attribute> = existing.to_vec();

    for new_attr in incoming {
        if let Some(existing_attr) = merged.iter_mut().find(|a| a.value == new_attr.value) {
            existing_attr.evidences.extend(new_attr.evidences.clone());
            let len = existing_attr.evidences.len();
            if len > MAX_EVIDENCES {
                existing_attr.evidences = existing_attr.evidences[len - MAX_EVIDENCES..].to_vec();
            }
        } else {
            merged.push(new_attr.clone());
        }
    }

    merged
}
```

- [ ] **Step 3: Implement `merge_profiles()` public function**

```rust
/// Merge an incoming profile into an existing one.
/// - Leveled attributes: keep highest level
/// - Non-leveled attributes: merge lists
/// - Evidences: truncate to 10, keep most recent
pub fn merge_profiles(existing: &ProfileData, incoming: &ProfileData) -> ProfileData {
    ProfileData {
        user_name: incoming.user_name.clone().or(existing.user_name.clone()),
        hard_skills: merge_leveled(&existing.hard_skills, &incoming.hard_skills),
        soft_skills: merge_leveled(&existing.soft_skills, &incoming.soft_skills),
        motivation_system: merge_leveled(&existing.motivation_system, &incoming.motivation_system),
        fear_system: merge_leveled(&existing.fear_system, &incoming.fear_system),
        value_system: merge_leveled(&existing.value_system, &incoming.value_system),
        humor_use: merge_leveled(&existing.humor_use, &incoming.humor_use),
        colloquialism: merge_leveled(&existing.colloquialism, &incoming.colloquialism),
        personality: merge_attributes(&existing.personality, &incoming.personality),
        way_of_decision_making: merge_attributes(&existing.way_of_decision_making, &incoming.way_of_decision_making),
        interests: merge_attributes(&existing.interests, &incoming.interests),
        user_goal: merge_attributes(&existing.user_goal, &incoming.user_goal),
        work_responsibility: merge_attributes(&existing.work_responsibility, &incoming.work_responsibility),
        working_habit_preference: merge_attributes(&existing.working_habit_preference, &incoming.working_habit_preference),
        tendency: merge_attributes(&existing.tendency, &incoming.tendency),
    }
}
```

- [ ] **Step 4: Add unit tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_leveled_keeps_highest_level() {
        let existing = vec![LeveledAttribute {
            value: "Rust".into(),
            level: "intermediate".into(),
            evidences: vec!["ep_1".into()],
        }];
        let incoming = vec![LeveledAttribute {
            value: "Rust".into(),
            level: "advanced".into(),
            evidences: vec!["ep_5".into()],
        }];
        let merged = merge_leveled(&existing, &incoming);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].level, "advanced");
        assert_eq!(merged[0].evidences, vec!["ep_1", "ep_5"]);
    }

    #[test]
    fn test_merge_leveled_does_not_downgrade() {
        let existing = vec![LeveledAttribute {
            value: "Rust".into(),
            level: "expert".into(),
            evidences: vec!["ep_1".into()],
        }];
        let incoming = vec![LeveledAttribute {
            value: "Rust".into(),
            level: "beginner".into(),
            evidences: vec!["ep_5".into()],
        }];
        let merged = merge_leveled(&existing, &incoming);
        assert_eq!(merged[0].level, "expert");
    }

    #[test]
    fn test_merge_adds_new_attributes() {
        let existing = vec![Attribute {
            value: "AI agents".into(),
            evidences: vec![],
        }];
        let incoming = vec![Attribute {
            value: "hardware hacking".into(),
            evidences: vec!["ep_3".into()],
        }];
        let merged = merge_attributes(&existing, &incoming);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_evidence_truncation() {
        let existing = vec![Attribute {
            value: "test".into(),
            evidences: (0..8).map(|i| format!("ep_{i}")).collect(),
        }];
        let incoming = vec![Attribute {
            value: "test".into(),
            evidences: (8..15).map(|i| format!("ep_{i}")).collect(),
        }];
        let merged = merge_attributes(&existing, &incoming);
        assert_eq!(merged[0].evidences.len(), MAX_EVIDENCES);
        // Should keep most recent (last 10)
        assert_eq!(merged[0].evidences[0], "ep_5");
    }

    #[test]
    fn test_merge_profiles_preserves_user_name() {
        let existing = ProfileData {
            user_name: Some("Fabio".into()),
            ..Default::default()
        };
        let incoming = ProfileData::default();
        let merged = merge_profiles(&existing, &incoming);
        assert_eq!(merged.user_name, Some("Fabio".into()));
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test --lib pipeline::merger`
Expected: all 5 tests pass.

- [ ] **Step 6: Add `pub mod merger;` to `mod.rs`**

- [ ] **Step 7: Commit**

```bash
git add src/memory/pipeline/merger.rs src/memory/pipeline/mod.rs
git commit -m "feat(pipeline): add ProfileMerger with leveled attribute handling"
```

---

## Chunk 2: Store + Config + ProfileExtractor

### Task 3: Add `profiles` table to PipelineStore

**Files:**
- Modify: `src/memory/pipeline/store.rs`

- [ ] **Step 1: Add profiles table creation to `init_schema()`**

Add after the `foresights` table creation:

```rust
conn.execute_batch(
    "CREATE TABLE IF NOT EXISTS profiles (
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
    CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id);",
)?;
```

- [ ] **Step 2: Add `get_profile()` method**

```rust
pub struct ProfileRow {
    pub id: String,
    pub user_id: String,
    pub profile_data: String,  // JSON string
    pub confidence: f64,
    pub version: i64,
    pub episode_count: i64,
    pub last_episode_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

pub async fn get_profile(&self, user_id: &str) -> Result<Option<ProfileRow>> {
    let conn = self.conn.lock().await;
    let mut stmt = conn.prepare_cached(
        "SELECT id, user_id, profile_data, confidence, version, episode_count,
                last_episode_id, created_at, updated_at
         FROM profiles WHERE user_id = ?1",
    )?;
    let result = stmt.query_row(params![user_id], |row| {
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
```

- [ ] **Step 3: Add `upsert_profile()` method**

```rust
pub async fn upsert_profile(
    &self,
    user_id: &str,
    profile_data_json: &str,
    confidence: f64,
    last_episode_id: &str,
) -> Result<()> {
    let conn = self.conn.lock().await;
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO profiles (id, user_id, profile_data, confidence, version, episode_count, last_episode_id, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, 1, 1, ?5, ?6, ?6)
         ON CONFLICT(user_id) DO UPDATE SET
           profile_data = ?3,
           confidence = ?4,
           version = version + 1,
           last_episode_id = ?5,
           updated_at = ?6",
        params![uuid::Uuid::new_v4().to_string(), user_id, profile_data_json, confidence, last_episode_id, now],
    )?;
    Ok(())
}
```

- [ ] **Step 4: Add `increment_episode_count()` method**

Returns the new count so the caller can check the trigger condition.

```rust
/// Increment episode_count for a user. Returns the new count.
/// Creates the profile row if it doesn't exist (with empty profile_data).
pub async fn increment_episode_count(&self, user_id: &str) -> Result<i64> {
    let conn = self.conn.lock().await;
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO profiles (id, user_id, profile_data, confidence, version, episode_count, created_at, updated_at)
         VALUES (?1, ?2, '{}', 0.0, 1, 1, ?3, ?3)
         ON CONFLICT(user_id) DO UPDATE SET
           episode_count = episode_count + 1,
           updated_at = ?3",
        params![uuid::Uuid::new_v4().to_string(), user_id, now],
    )?;
    let count: i64 = conn.query_row(
        "SELECT episode_count FROM profiles WHERE user_id = ?1",
        params![user_id],
        |row| row.get(0),
    )?;
    Ok(count)
}
```

- [ ] **Step 5: Add `get_recent_episodes()` method**

Retrieves the last N episodes for a given user_id, to pass to the profile extractor.

```rust
pub async fn get_recent_episodes(&self, user_id: &str, limit: usize) -> Result<Vec<EpisodeRow>> {
    let conn = self.conn.lock().await;
    let mut stmt = conn.prepare_cached(
        "SELECT id, session_id, user_id, subject, summary, episode, created_at
         FROM episodes
         WHERE user_id = ?1
         ORDER BY created_at DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![user_id, limit as i64], |row| {
        Ok(EpisodeRow {
            id: row.get(0)?,
            session_id: row.get(1)?,
            user_id: row.get(2)?,
            subject: row.get(3)?,
            summary: row.get(4)?,
            episode: row.get(5)?,
            created_at: row.get(6)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}
```

**Note**: `EpisodeRow` should already exist from Phase 1. If the struct is defined in `store.rs`, reuse it. If not, define it here alongside `ProfileRow`.

- [ ] **Step 6: Compile and verify**

Run: `cargo check`

- [ ] **Step 7: Commit**

```bash
git add src/memory/pipeline/store.rs
git commit -m "feat(pipeline): add profiles table and store methods"
```

---

### Task 4: Add `profile_update_interval` to config

**Files:**
- Modify: `src/memory/pipeline/config.rs`
- Modify: `src/config/schema.rs`

- [ ] **Step 1: Add field to `PipelineConfig`**

```rust
pub profile_update_interval: usize,
```

Default:
```rust
profile_update_interval: 5,
```

- [ ] **Step 2: Wire to top-level config in `schema.rs`**

Add `profile_update_interval: Option<usize>` to the pipeline config section in `schema.rs` (same pattern as `max_foresight_entries` from Phase 2).

- [ ] **Step 3: Compile and verify**

Run: `cargo check`

- [ ] **Step 4: Commit**

```bash
git add src/memory/pipeline/config.rs src/config/schema.rs
git commit -m "feat(config): add profile_update_interval to pipeline config (default 5)"
```

---

### Task 5: Create profile extraction prompts

**Files:**
- Create: `src/memory/pipeline/prompts/en/profile.txt`
- Create: `src/memory/pipeline/prompts/it/profile.txt`

- [ ] **Step 1: Create English profile prompt**

```text
Analyze these conversation episodes and extract/update the user profile.

Current profile (may be empty if first extraction):
{current_profile_json}

New episodes to analyze:
{recent_episodes}

For each attribute, provide evidence (episode IDs that support it).
For skill-like attributes, assign a level: "beginner", "intermediate", "advanced", "expert".

Respond ONLY with valid JSON matching this schema:
{
  "user_name": "string or null",
  "hard_skills": [{"value": "skill name", "level": "level", "evidences": ["ep_id"]}],
  "soft_skills": [{"value": "skill name", "level": "level", "evidences": ["ep_id"]}],
  "personality": [{"value": "trait description", "evidences": ["ep_id"]}],
  "way_of_decision_making": [{"value": "description", "evidences": ["ep_id"]}],
  "interests": [{"value": "interest", "evidences": ["ep_id"]}],
  "user_goal": [{"value": "goal", "evidences": ["ep_id"]}],
  "work_responsibility": [{"value": "responsibility", "evidences": ["ep_id"]}],
  "working_habit_preference": [{"value": "preference", "evidences": ["ep_id"]}],
  "tendency": [{"value": "tendency", "evidences": ["ep_id"]}],
  "motivation_system": [{"value": "motivation", "level": "level", "evidences": ["ep_id"]}],
  "fear_system": [{"value": "fear", "level": "level", "evidences": ["ep_id"]}],
  "value_system": [{"value": "value", "level": "level", "evidences": ["ep_id"]}],
  "humor_use": [{"value": "humor style", "level": "level", "evidences": ["ep_id"]}],
  "colloquialism": [{"value": "speech pattern", "level": "level", "evidences": ["ep_id"]}]
}

Rules:
- Merge with existing profile: update levels if evidence is stronger, add new attributes
- Keep highest level when conflicting evidence exists
- Truncate evidences to max 10 per attribute
- Only add attributes with clear evidence, never speculate
- Omit empty arrays (attributes with no evidence)
```

- [ ] **Step 2: Create Italian profile prompt**

Same structure as the English prompt, but **all instructions and descriptions are in Italian**. The JSON field names (keys) MUST remain in English exactly as they appear in `ProfileData` — only translate the natural-language instructions, not the attribute names.

Example: the JSON schema in the prompt must still use `"hard_skills"`, `"soft_skills"`, `"personality"`, etc. — NOT `"competenze_tecniche"` or other Italian translations. Translating the JSON keys would break `ProfileData` deserialization.

Correct pattern:
```text
Analizza questi episodi di conversazione ed estrai/aggiorna il profilo utente.

Profilo attuale (può essere vuoto se è la prima estrazione):
{current_profile_json}

Nuovi episodi da analizzare:
{recent_episodes}

Per ogni attributo, fornisci le evidenze (ID degli episodi che lo supportano).
Per gli attributi con livello, assegna uno tra: "beginner", "intermediate", "advanced", "expert".

Rispondi SOLO con JSON valido con questa struttura (le chiavi JSON devono restare in inglese):
{
  "user_name": "...",
  "hard_skills": [{"value": "...", "level": "...", "evidences": ["ep_id"]}],
  "soft_skills": [...],
  "personality": [{"value": "...", "evidences": ["ep_id"]}],
  ...
}

Regole:
- Integra con il profilo esistente: aggiorna i livelli se le evidenze sono più forti, aggiungi nuovi attributi
- Mantieni il livello più alto in caso di evidenze contrastanti
- Tronca le evidenze a max 10 per attributo
- Aggiungi attributi solo con evidenze chiare, mai speculare
```

- [ ] **Step 3: Commit**

```bash
git add src/memory/pipeline/prompts/en/profile.txt src/memory/pipeline/prompts/it/profile.txt
git commit -m "feat(pipeline): add profile extraction prompts (en, it)"
```

---

### Task 6: Create ProfileExtractor

**Files:**
- Create: `src/memory/pipeline/extractor/profile.rs`
- Modify: `src/memory/pipeline/extractor/mod.rs`

- [ ] **Step 1: Implement `extract()` function**

```rust
// src/memory/pipeline/extractor/profile.rs
use anyhow::Result;
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};
use super::super::profile_types::ProfileData;
use super::super::store::EpisodeRow;

const PROMPT_EN: &str = include_str!("../prompts/en/profile.txt");
const PROMPT_IT: &str = include_str!("../prompts/it/profile.txt");

pub async fn extract(
    provider: &dyn Provider,
    model: &str,
    current_profile: &ProfileData,
    recent_episodes: &[EpisodeRow],
    prompt_language: &str,
) -> Result<ProfileData> {
    let template = match prompt_language {
        "it" => PROMPT_IT,
        _ => PROMPT_EN,
    };

    let current_json = serde_json::to_string_pretty(current_profile)?;

    let episodes_text = recent_episodes
        .iter()
        .map(|ep| format!(
            "[Episode {}] ({})\nSubject: {}\n{}\n",
            ep.id,
            ep.created_at,
            ep.subject.as_deref().unwrap_or("untitled"),
            ep.episode,
        ))
        .collect::<Vec<_>>()
        .join("\n---\n");

    let prompt = template
        .replace("{current_profile_json}", &current_json)
        .replace("{recent_episodes}", &episodes_text);

    let messages = [
        ChatMessage::system("You are a user profile extraction engine. Analyze episodes and extract structured attributes."),
        ChatMessage::user(&prompt),
    ];
    let request = ChatRequest {
        messages: &messages,
        tools: None,
    };

    let response = provider.chat(request, model, 0.1).await?;
    let text = response.text.unwrap_or_default();
    let text = text.trim();

    let parsed: ProfileData = serde_json::from_str(text)
        .map_err(|e| anyhow::anyhow!("Profile JSON parse error: {e}\nRaw: {text}"))?;

    Ok(parsed)
}
```

**Note**: Uses the same `ChatRequest` + `ChatResponse.text` pattern as all Phase 1/2 extractors.

- [ ] **Step 2: Add `pub mod profile;` to extractor/mod.rs**

```rust
pub mod profile;
```

- [ ] **Step 3: Compile and verify**

Run: `cargo check`

- [ ] **Step 4: Commit**

```bash
git add src/memory/pipeline/extractor/profile.rs src/memory/pipeline/extractor/mod.rs
git commit -m "feat(pipeline): add ProfileExtractor with LLM extraction"
```

---

## Chunk 3: Orchestration + Formatter Integration

### Task 7: Update orchestration with profile trigger

**Files:**
- Modify: `src/memory/pipeline/mod.rs`

- [ ] **Step 1: Add profile trigger logic to `process_turn()`**

After saving episode + event_logs + foresights (Phase 1+2 code), add:

```rust
// --- Profile trigger (Phase 3) ---
// Design decision: increment episode_count AFTER saving episode+event_logs+foresights
// (not in parallel with foresight as the spec's data flow diagram suggests).
// Trade-off: if episode save fails and process_turn_inner returns early,
// the counter is never incremented — no off-by-one risk.
// If save succeeds but counter increment fails, the profile trigger may be
// delayed by one episode — acceptable for a best-effort feature.
let episode_count = match self.store.increment_episode_count(user_id).await {
    Ok(count) => count,
    Err(e) => {
        tracing::warn!("Failed to increment episode count (non-blocking): {e}");
        0
    }
};

if episode_count > 0
    && episode_count as usize % self.config.profile_update_interval == 0
{
    // Gather recent episodes for this user
    let recent_episodes = self
        .store
        .get_recent_episodes(user_id, self.config.profile_update_interval)
        .await
        .unwrap_or_default();

    if !recent_episodes.is_empty() {
        // Load current profile
        let current_profile = match self.store.get_profile(user_id).await {
            Ok(Some(row)) => serde_json::from_str::<ProfileData>(&row.profile_data)
                .unwrap_or_default(),
            _ => ProfileData::default(),
        };

        // Extract new profile from LLM
        let extraction_model = self
            .config
            .extraction_model
            .as_deref()
            .unwrap_or(default_model);

        match profile::extract(
            provider,
            extraction_model,
            &current_profile,
            &recent_episodes,
            &self.config.prompt_language,
        )
        .await
        {
            Ok(new_profile) => {
                let merged = merger::merge_profiles(&current_profile, &new_profile);
                let merged_json = serde_json::to_string(&merged).unwrap_or_default();
                if let Err(e) = self
                    .store
                    .upsert_profile(user_id, &merged_json, 0.0, &episode_id)
                    .await
                {
                    tracing::warn!("Failed to save profile (non-blocking): {e}");
                }
            }
            Err(e) => {
                tracing::warn!("Profile extraction failed (non-blocking): {e}");
            }
        }
    }
}
```

**Key**: Profile extraction is entirely non-blocking. If anything fails, the rest of the pipeline (episode, event_log, foresight) is unaffected.

- [ ] **Step 2: Compile and verify**

Run: `cargo check`

- [ ] **Step 3: Commit**

```bash
git add src/memory/pipeline/mod.rs
git commit -m "feat(pipeline): integrate profile trigger into process_turn orchestration"
```

---

### Task 8: Update Formatter to include `<trait>` section

**Files:**
- Modify: `src/memory/pipeline/formatter.rs`

- [ ] **Step 1: Add `render_trait()` helper**

Converts `ProfileData` JSON into a human-readable summary for the `<trait>` section.

```rust
use super::profile_types::ProfileData;

fn render_trait(profile: &ProfileData) -> String {
    // Renders a human-readable subset of ProfileData for the <trait> XML section.
    // Included: user_name, hard_skills+soft_skills (combined), personality, user_goal,
    //           working_habit_preference+tendency (as "Preferences"), interests.
    // Omitted: motivation_system, fear_system, value_system, humor_use,
    //          colloquialism, way_of_decision_making, work_responsibility.
    // These omitted fields can be added in a future revision if the LLM needs them.
    let mut lines = Vec::new();

    if let Some(name) = &profile.user_name {
        lines.push(format!("    Name: {name}"));
    }

    // Skills: combine hard + soft with levels
    let skills: Vec<String> = profile
        .hard_skills
        .iter()
        .chain(profile.soft_skills.iter())
        .map(|s| format!("{} ({})", s.value, s.level))
        .collect();
    if !skills.is_empty() {
        lines.push(format!("    Skills: {}", skills.join(", ")));
    }

    // Personality
    let traits: Vec<&str> = profile.personality.iter().map(|a| a.value.as_str()).collect();
    if !traits.is_empty() {
        lines.push(format!("    Personality: {}", traits.join(", ")));
    }

    // Goals
    let goals: Vec<&str> = profile.user_goal.iter().map(|a| a.value.as_str()).collect();
    if !goals.is_empty() {
        lines.push(format!("    Goals: {}", goals.join(", ")));
    }

    // Preferences (working_habit + tendency)
    let prefs: Vec<&str> = profile
        .working_habit_preference
        .iter()
        .chain(profile.tendency.iter())
        .map(|a| a.value.as_str())
        .collect();
    if !prefs.is_empty() {
        lines.push(format!("    Preferences: {}", prefs.join(", ")));
    }

    // Interests
    let interests: Vec<&str> = profile.interests.iter().map(|a| a.value.as_str()).collect();
    if !interests.is_empty() {
        lines.push(format!("    Interests: {}", interests.join(", ")));
    }

    lines.join("\n")
}
```

- [ ] **Step 2: Add `<trait>` section to `build_xml_context()`**

Insert **before** the `<episodic>` section (first element in `<memory>` block):

```rust
// Trait section — user profile (Phase 3)
if let Ok(Some(profile_row)) = store.get_profile(user_id).await {
    if let Ok(profile) = serde_json::from_str::<ProfileData>(&profile_row.profile_data) {
        let trait_text = render_trait(&profile);
        if !trait_text.is_empty() {
            xml.push_str("  <trait>\n");
            xml.push_str(&trait_text);
            xml.push_str("\n  </trait>\n");
        }
    }
}
```

- [ ] **Step 3: Verify `build_xml_context()` has access to `user_id`**

The formatter needs `user_id` to query the profile. If `build_xml_context()` doesn't already receive `user_id` as a parameter, add it. This was already part of Phase 1 design (the function needs it for profile injection):

```rust
pub async fn build_xml_context(
    store: &PipelineStore,
    memory: &dyn Memory,
    query: &str,
    user_id: &str,      // needed for profile lookup
    config: &PipelineConfig,
) -> Result<String> {
```

- [ ] **Step 4: Compile and verify**

Run: `cargo check`

- [ ] **Step 5: Verify expected XML output**

Full XML with all 4 sections:

```text
<memory>
  <trait>
    Name: Fabio Biola
    Skills: Rust (advanced), Python (intermediate)
    Personality: cautious, prefers understanding before acting
    Goals: build Nexus AI assistant on Bison Pro
    Preferences: Italian communication, brainstorm before implementing
    Interests: AI agents, hardware hacking
  </trait>
  <episodic>
    [1] (2026-04-05) ...
  </episodic>
  <facts>
    - ...
  </facts>
  <foresight>
    - ...
  </foresight>
</memory>
```

- [ ] **Step 6: Commit**

```bash
git add src/memory/pipeline/formatter.rs
git commit -m "feat(pipeline): add <trait> section to XML injection"
```

---

### Task 9: Integration test

**Files:**
- No new test files

- [ ] **Step 1: Build the binary**

Run: `cargo build --release`

- [ ] **Step 2: Verify schema migration**

Start ZeroClaw with pipeline enabled. Check that `profiles` table exists:

```bash
ssh bison-pro "python3 -c \"
import sqlite3
conn = sqlite3.connect('/home/nexus/workspace/memory/brain.db')
cursor = conn.execute(\\\"SELECT name FROM sqlite_master WHERE type='table' AND name='profiles'\\\")
print('profiles table:', 'EXISTS' if cursor.fetchone() else 'MISSING')
conn.close()
\""
```

- [ ] **Step 3: Test profile extraction trigger**

Send enough messages via Telegram to generate 5 episodes (the default `profile_update_interval`). After the 5th boundary flush, check logs:

```bash
ssh bison-pro "pm2 logs zeroclaw-gateway --lines 50 --nostream | grep -i profile"
```

Expected: a log line about profile extraction running.

- [ ] **Step 4: Verify profile storage**

```bash
ssh bison-pro "python3 -c \"
import sqlite3, json
conn = sqlite3.connect('/home/nexus/workspace/memory/brain.db')
row = conn.execute('SELECT user_id, profile_data, version, episode_count FROM profiles LIMIT 1').fetchone()
if row:
    print(f'user_id: {row[0]}')
    print(f'version: {row[2]}, episode_count: {row[3]}')
    print(json.dumps(json.loads(row[1]), indent=2))
else:
    print('No profiles yet')
conn.close()
\""
```

- [ ] **Step 5: Verify XML injection includes trait section**

Check logs for the XML block containing `<trait>`. Look for rendered profile attributes.

- [ ] **Step 6: Final commit (if any fixes needed)**

```bash
git add src/memory/pipeline/
git commit -m "fix(pipeline): profile integration fixes from testing"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `profile_types.rs` | `ProfileData`, `LeveledAttribute`, `Attribute` serde structs |
| 2 | `merger.rs` | `ProfileMerger` with leveled merge + evidence truncation + unit tests |
| 3 | `store.rs` | `profiles` table + `get_profile()` + `upsert_profile()` + `increment_episode_count()` + `get_recent_episodes()` |
| 4 | `config.rs`, `schema.rs` | `profile_update_interval` (default 5) |
| 5 | `prompts/en/profile.txt`, `prompts/it/profile.txt` | Profile extraction prompts |
| 6 | `extractor/profile.rs`, `extractor/mod.rs` | `ProfileExtractor` with LLM call + JSON parse |
| 7 | `mod.rs` | Orchestration: conditional profile trigger every N episodes |
| 8 | `formatter.rs` | `<trait>` section as first element in XML block |
| 9 | — | Integration test on device |
