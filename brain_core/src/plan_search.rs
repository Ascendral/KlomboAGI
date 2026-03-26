use crate::retrieval::token_set;
use crate::types::{PlanCandidate, PlanScore};

pub fn score_candidates(
    goal: &str,
    candidates: &[PlanCandidate],
    anti_patterns: &[String],
) -> Vec<PlanScore> {
    let goal_tokens = token_set(goal);

    candidates
        .iter()
        .map(|candidate| {
            let candidate_tokens = token_set(&candidate.description);
            let overlap = goal_tokens.intersection(&candidate_tokens).count() as f32;
            let candidate_phrase = normalize_phrase(&candidate.description);
            let anti_pattern_penalty = anti_patterns.iter().fold(0.0, |penalty, pattern| {
                let pattern_tokens = token_set(pattern);
                let token_overlap = candidate_tokens.intersection(&pattern_tokens).count() as f32;
                let phrase_bonus = if !candidate_phrase.is_empty()
                    && normalize_phrase(pattern).contains(&candidate_phrase)
                {
                    2.0
                } else {
                    0.0
                };
                penalty + (token_overlap * 0.25) + phrase_bonus
            });

            let score = overlap - candidate.estimated_cost - anti_pattern_penalty;
            let reasons = vec![
                format!("goal_overlap={overlap:.2}"),
                format!("estimated_cost={:.2}", candidate.estimated_cost),
                format!("anti_pattern_penalty={anti_pattern_penalty:.2}"),
            ];

            PlanScore {
                candidate_id: candidate.id.clone(),
                score,
                reasons,
            }
        })
        .collect()
}

fn normalize_phrase(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}
