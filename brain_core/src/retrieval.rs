use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::types::RetrievalHit;

pub fn rank_memories(query: &str, memories: &[String], limit: usize) -> Vec<RetrievalHit> {
    let query_tokens = token_set(query);
    let mut hits: Vec<RetrievalHit> = memories
        .iter()
        .filter_map(|memory| {
            let memory_tokens = token_set(memory);
            let overlap = query_tokens.intersection(&memory_tokens).count() as f32;
            if overlap <= 0.0 {
                return None;
            }

            let denominator = query_tokens.len().max(memory_tokens.len()) as f32;
            Some(RetrievalHit {
                memory: memory.clone(),
                score: overlap / denominator,
            })
        })
        .collect();

    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });

    if limit > 0 && hits.len() > limit {
        hits.truncate(limit);
    }
    hits
}

pub(crate) fn token_set(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}
