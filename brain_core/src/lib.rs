use pyo3::prelude::*;

mod learning;
mod plan_search;
mod retrieval;
mod scoring;
mod transfer;
mod types;

use types::{PlanCandidate, TrajectorySummary};

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

#[pyfunction]
fn retrieve_memory(query: String, memories: Vec<String>, limit: usize) -> Vec<(String, f32)> {
    retrieval::rank_memories(&query, &memories, limit)
        .into_iter()
        .map(|hit| (hit.memory, hit.score))
        .collect()
}

#[pyfunction]
fn score_plan(
    goal: String,
    candidates: Vec<(String, String, String, f32)>,
    anti_patterns: Vec<String>,
) -> Vec<(String, f32, Vec<String>)> {
    let plan_candidates = candidates
        .into_iter()
        .map(|(id, description, action_kind, estimated_cost)| PlanCandidate {
            id,
            description,
            action_kind,
            estimated_cost,
        })
        .collect::<Vec<PlanCandidate>>();

    plan_search::score_candidates(&goal, &plan_candidates, &anti_patterns)
        .into_iter()
        .map(|score| (score.candidate_id, score.score, score.reasons))
        .collect()
}

#[pyfunction]
fn transfer_score(source_tags: Vec<String>, target_tags: Vec<String>) -> (f32, Vec<String>, Vec<String>) {
    let result = transfer::transfer_overlap(&source_tags, &target_tags);
    (result.overlap, result.shared_tags, result.missing_tags)
}

#[pyfunction]
fn reliability_score(successes: u32, failures: u32) -> f32 {
    scoring::reliability_score(successes, failures)
}

#[pyfunction]
#[pyo3(signature = (task_id, domain, actions, success, failure_reason=None))]
fn extract_learning(
    task_id: String,
    domain: String,
    actions: Vec<String>,
    success: bool,
    failure_reason: Option<String>,
) -> Option<(String, Vec<String>, f32)> {
    let summary = TrajectorySummary {
        task_id,
        domain,
        actions,
        success,
        failure_reason,
    };

    learning::extract_learning(&summary)
        .map(|procedure| (procedure.name, procedure.steps, procedure.confidence))
}

#[pyfunction]
#[pyo3(signature = (task_id, domain, actions, failure_reason=None))]
fn anti_pattern(
    task_id: String,
    domain: String,
    actions: Vec<String>,
    failure_reason: Option<String>,
) -> Option<String> {
    let summary = TrajectorySummary {
        task_id,
        domain,
        actions,
        success: false,
        failure_reason,
    };
    learning::anti_pattern(&summary)
}

#[pyfunction]
#[pyo3(signature = (task_id, domain, actions, success, failure_reason=None))]
fn trajectory_outcome_score(
    task_id: String,
    domain: String,
    actions: Vec<String>,
    success: bool,
    failure_reason: Option<String>,
) -> f32 {
    let summary = TrajectorySummary {
        task_id,
        domain,
        actions,
        success,
        failure_reason,
    };
    scoring::trajectory_outcome_score(&summary)
}

#[pymodule]
fn brain_core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(retrieve_memory, module)?)?;
    module.add_function(wrap_pyfunction!(score_plan, module)?)?;
    module.add_function(wrap_pyfunction!(transfer_score, module)?)?;
    module.add_function(wrap_pyfunction!(reliability_score, module)?)?;
    module.add_function(wrap_pyfunction!(extract_learning, module)?)?;
    module.add_function(wrap_pyfunction!(anti_pattern, module)?)?;
    module.add_function(wrap_pyfunction!(trajectory_outcome_score, module)?)?;
    Ok(())
}
