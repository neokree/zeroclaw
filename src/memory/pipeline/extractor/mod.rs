// src/memory/pipeline/extractor/mod.rs
pub mod episode;
pub mod event_log;
pub mod foresight;
pub mod profile;

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
    pub foresights: Vec<foresight::ForesightEntry>,
}
