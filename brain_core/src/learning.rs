use crate::types::{LearnedProcedure, TrajectorySummary};

pub fn extract_learning(summary: &TrajectorySummary) -> Option<LearnedProcedure> {
    if !summary.success || summary.actions.is_empty() {
        return None;
    }

    Some(LearnedProcedure {
        name: format!("Procedure for {}", summary.task_id),
        steps: summary.actions.clone(),
        confidence: 0.5 + (summary.actions.len() as f32 * 0.05),
    })
}

pub fn anti_pattern(summary: &TrajectorySummary) -> Option<String> {
    if summary.success {
        return None;
    }

    Some(
        summary
            .failure_reason
            .clone()
            .unwrap_or_else(|| format!("Avoid repeating failed trajectory {}", summary.task_id)),
    )
}
