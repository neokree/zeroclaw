// src/memory/pipeline/profile_types.rs
use serde::{Deserialize, Serialize};

/// Attribute with skill level (hard_skills, soft_skills, motivation_system, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeveledAttribute {
    pub value: String,
    pub level: String, // "beginner" | "intermediate" | "advanced" | "expert"
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
