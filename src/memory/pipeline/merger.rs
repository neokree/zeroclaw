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
