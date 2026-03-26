use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalHit {
    pub memory: String,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanCandidate {
    pub id: String,
    pub description: String,
    pub action_kind: String,
    pub estimated_cost: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanScore {
    pub candidate_id: String,
    pub score: f32,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferMatch {
    pub overlap: f32,
    pub shared_tags: Vec<String>,
    pub missing_tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectorySummary {
    pub task_id: String,
    pub domain: String,
    pub actions: Vec<String>,
    pub success: bool,
    pub failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnedProcedure {
    pub name: String,
    pub steps: Vec<String>,
    pub confidence: f32,
}
