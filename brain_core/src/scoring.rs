use crate::types::TrajectorySummary;

pub fn reliability_score(successes: u32, failures: u32) -> f32 {
    let total = successes + failures;
    if total == 0 {
        return 0.0;
    }
    successes as f32 / total as f32
}

pub fn trajectory_outcome_score(summary: &TrajectorySummary) -> f32 {
    let action_bonus = summary.actions.len() as f32 * 0.05;
    if summary.success {
        1.0 + action_bonus
    } else {
        0.0 - action_bonus
    }
}
