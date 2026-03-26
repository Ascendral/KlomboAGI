//! brain_core — Fast, deterministic cognition kernel for KlomboAGI.
//!
//! Python stays the control plane. Rust is the fast path for:
//! - Memory retrieval and ranking
//! - Plan search and scoring
//! - Transfer matching between episodes
//! - Skill/failure-pattern extraction
//! - World-state updates
//!
//! Exposed to Python via pyo3.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Knowledge Graph ──

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Belief {
    subject: String,
    predicate: String,
    object: String,
    frequency: f64,    // NARS truth value: how often true
    confidence: f64,   // NARS truth value: how much evidence
    source: String,    // "human", "deduction", "search"
    timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryEntry {
    id: String,
    content: String,
    domain: String,
    relevance_score: f64,
    use_count: u32,
    success_count: u32,
    failure_count: u32,
}

// ── Core Functions ──

#[pyfunction]
fn retrieve_memory(goal: &str, _state: &str, memories_json: &str) -> PyResult<String> {
    let memories: Vec<MemoryEntry> = serde_json::from_str(memories_json)
        .unwrap_or_default();

    let goal_words: Vec<&str> = goal.split_whitespace().collect();

    let mut scored: Vec<(f64, &MemoryEntry)> = memories.iter()
        .map(|m| {
            let content_words: Vec<&str> = m.content.split_whitespace().collect();
            let overlap = goal_words.iter()
                .filter(|w| content_words.contains(w))
                .count() as f64;

            let reliability = if m.use_count > 0 {
                m.success_count as f64 / m.use_count as f64
            } else {
                0.5
            };

            let score = overlap * reliability * m.relevance_score;
            (score, m)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let top: Vec<&MemoryEntry> = scored.iter()
        .take(5)
        .filter(|(score, _)| *score > 0.0)
        .map(|(_, m)| *m)
        .collect();

    Ok(serde_json::to_string(&top).unwrap_or_default())
}

#[pyfunction]
fn score_plan(plan_json: &str, failures_json: &str, skills_json: &str) -> PyResult<f64> {
    let plan: HashMap<String, serde_json::Value> = serde_json::from_str(plan_json)
        .unwrap_or_default();
    let failures: Vec<HashMap<String, String>> = serde_json::from_str(failures_json)
        .unwrap_or_default();
    let skills: Vec<HashMap<String, serde_json::Value>> = serde_json::from_str(skills_json)
        .unwrap_or_default();

    let mut score: f64 = 0.5; // Base score

    // Boost if a matching skill exists
    let plan_desc = plan.get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    for skill in &skills {
        if let Some(desc) = skill.get("description").and_then(|v| v.as_str()) {
            let overlap = plan_desc.split_whitespace()
                .filter(|w| desc.contains(w))
                .count();
            if overlap > 2 {
                score += 0.2;
            }
        }
    }

    // Penalize if plan matches a known failure pattern
    for failure in &failures {
        if let Some(action) = failure.get("failed_action") {
            if plan_desc.contains(action.as_str()) {
                score -= 0.3;
            }
        }
    }

    Ok(score.max(0.0).min(1.0))
}

#[pyfunction]
fn transfer_match(source_json: &str, target_description: &str) -> PyResult<f64> {
    let source: HashMap<String, serde_json::Value> = serde_json::from_str(source_json)
        .unwrap_or_default();

    let source_desc = source.get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let source_words: Vec<&str> = source_desc.split_whitespace().collect();
    let target_words: Vec<&str> = target_description.split_whitespace().collect();

    if source_words.is_empty() || target_words.is_empty() {
        return Ok(0.0);
    }

    let overlap = source_words.iter()
        .filter(|w| target_words.contains(w))
        .count() as f64;

    let max_len = source_words.len().max(target_words.len()) as f64;

    Ok(overlap / max_len)
}

#[pyfunction]
fn nars_revision(f1: f64, c1: f64, f2: f64, c2: f64) -> PyResult<(f64, f64)> {
    let k = 1.0; // Horizon parameter
    let w1 = k * c1 / (1.0 - c1 + f64::EPSILON);
    let w2 = k * c2 / (1.0 - c2 + f64::EPSILON);
    let w = w1 + w2;
    let f = if w > 0.0 { (w1 * f1 + w2 * f2) / w } else { 0.5 };
    let c = w / (w + k);
    Ok((f, c))
}

#[pyfunction]
fn nars_deduction(f1: f64, c1: f64, f2: f64, c2: f64) -> PyResult<(f64, f64)> {
    Ok((f1 * f2, f1 * f2 * c1 * c2))
}

// ── Python Module ──

#[pymodule]
fn brain_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(retrieve_memory, m)?)?;
    m.add_function(wrap_pyfunction!(score_plan, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_match, m)?)?;
    m.add_function(wrap_pyfunction!(nars_revision, m)?)?;
    m.add_function(wrap_pyfunction!(nars_deduction, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nars_revision() {
        let (f, c) = nars_revision(1.0, 0.5, 1.0, 0.5).unwrap();
        assert!(c > 0.5); // More evidence = higher confidence
        assert!((f - 1.0).abs() < 0.01); // Both positive = positive
    }

    #[test]
    fn test_nars_deduction() {
        let (f, c) = nars_deduction(0.9, 0.9, 1.0, 0.95).unwrap();
        assert!((f - 0.9).abs() < 0.01);
        assert!(c < 0.9); // Chain weakens confidence
    }

    #[test]
    fn test_transfer_match() {
        let source = r#"{"description": "fix a bug in the login function"}"#;
        let target = "fix a bug in the payment function";
        let score = transfer_match(source, target).unwrap();
        assert!(score > 0.3); // "fix a bug in the" overlaps
    }

    #[test]
    fn test_score_plan() {
        let plan = r#"{"description": "search for login bug"}"#;
        let failures = r#"[{"failed_action": "deploy"}]"#;
        let skills = r#"[{"description": "search for patterns in code"}]"#;
        let score = score_plan(plan, failures, skills).unwrap();
        assert!(score >= 0.5); // Has matching skill, no matching failure
    }
}
