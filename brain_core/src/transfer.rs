use crate::retrieval::token_set;
use crate::types::TransferMatch;

pub fn transfer_overlap(source_tags: &[String], target_tags: &[String]) -> TransferMatch {
    let source = source_tags.iter().cloned().collect::<Vec<String>>().join(" ");
    let target = target_tags.iter().cloned().collect::<Vec<String>>().join(" ");
    let source_tokens = token_set(&source);
    let target_tokens = token_set(&target);

    let shared_tags = source_tokens
        .intersection(&target_tokens)
        .cloned()
        .collect::<Vec<String>>();
    let missing_tags = target_tokens
        .difference(&source_tokens)
        .cloned()
        .collect::<Vec<String>>();

    let denominator = source_tokens.len().max(target_tokens.len()) as f32;
    let overlap = if denominator == 0.0 {
        0.0
    } else {
        shared_tags.len() as f32 / denominator
    };

    TransferMatch {
        overlap,
        shared_tags,
        missing_tags,
    }
}
