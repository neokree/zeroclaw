// src/memory/pipeline/extractor/profile.rs
use anyhow::Result;
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};
use super::super::profile_types::ProfileData;
use super::super::store::SavedEpisode;

const PROMPT_EN: &str = include_str!("../prompts/en/profile.txt");
const PROMPT_IT: &str = include_str!("../prompts/it/profile.txt");

pub async fn extract(
    provider: &dyn Provider,
    model: &str,
    current_profile: &ProfileData,
    recent_episodes: &[SavedEpisode],
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
            ep.subject,
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
