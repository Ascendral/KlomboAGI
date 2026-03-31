"""
ARC Strategy Routing Classifier — predicts which strategy family will
solve a given ARC task based on extracted features.

Training workflow:
  1. Run collect_labels() to instrument the solver and record which
     strategy family solves each task
  2. Run train_classifier() to fit a RandomForest on features → labels
  3. Use predict_family() at inference time to route tasks

The classifier is non-destructive: if the predicted family fails,
the solver falls through to the existing pipeline.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from klomboagi.reasoning.arc_features import extract_feature_vector, feature_names

# Strategy family labels
FAMILIES = [
    "phase0",       # Phase 0 specific learners
    "phase1",       # Hand-coded strategies (SmartARCSolver base)
    "cell_rule",    # learn_cell_rule
    "object_rule",  # learn_object_rule
    "tiling",       # learn_tiling_rule
    "extraction",   # learn_extraction_rule
    "grid_ops",     # learn_grid_rule
    "region",       # learn_region_rule
    "gravity",      # learn_gravity_rule
    "advanced",     # learn_advanced_rule (symmetry, propagation, etc.)
    "context",      # learn_context_rule
    "ranking",      # learn_ranking_rule
    "legend",       # learn_legend_rule
    "compose",      # learn_compose_rule
    "multiobj",     # learn_multiobj_rule
    "pattern",      # learn_pattern_rule
    "dsl",          # DSL synthesis
    "none",         # No strategy works
]

MODEL_PATH = Path(__file__).parent / "arc_classifier_model.pkl"
LABELS_PATH = Path(__file__).parent / "arc_classifier_labels.json"


def collect_labels(tasks: list[dict] | None = None) -> list[dict]:
    """
    Run the solver with instrumentation to record which strategy family
    solves each task. Returns list of {task_id, features, label, correct}.

    Args:
        tasks: list of {id, train, test_input, test_output} dicts.
               If None, loads from arckit.
    """
    if tasks is None:
        tasks = _load_arckit_tasks()

    from klomboagi.reasoning.arc_cell_rules import (
        learn_span_fill_rule, learn_color_key_swap, learn_template_row_stamp,
        learn_grid_gap_fill, learn_single_cell_paint, learn_connect_dot_pairs,
        learn_cross_from_dots, learn_diamond_expand, learn_arrow_ray,
        learn_lshape_concavity, learn_conditional_span_fill,
    )
    from klomboagi.reasoning.arc_cell_rules import learn_cell_rule
    from klomboagi.reasoning.arc_object_rules import learn_object_rule
    from klomboagi.reasoning.arc_pattern_match import learn_pattern_rule
    from klomboagi.reasoning.arc_extraction import learn_extraction_rule
    from klomboagi.reasoning.arc_grid_ops import learn_grid_rule
    from klomboagi.reasoning.arc_region import learn_region_rule
    from klomboagi.reasoning.arc_gravity import learn_gravity_rule
    from klomboagi.reasoning.arc_advanced import learn_advanced_rule
    from klomboagi.reasoning.arc_tiling import learn_tiling_rule
    from klomboagi.reasoning.arc_context_rules import learn_context_rule
    from klomboagi.reasoning.arc_ranking import learn_ranking_rule
    from klomboagi.reasoning.arc_legend import learn_legend_rule
    from klomboagi.reasoning.arc_compose import learn_compose_rule
    from klomboagi.reasoning.arc_multiobj import learn_multiobj_rule
    from klomboagi.reasoning.arc_smart_solver import SmartARCSolverV2

    # Phase 0 learners with their labels
    phase0_learners = [
        (learn_span_fill_rule, "phase0"),
        (learn_conditional_span_fill, "phase0"),
        (learn_color_key_swap, "phase0"),
        (learn_template_row_stamp, "phase0"),
        (learn_grid_gap_fill, "phase0"),
        (learn_single_cell_paint, "phase0"),
        (learn_connect_dot_pairs, "phase0"),
        (learn_cross_from_dots, "phase0"),
        (learn_diamond_expand, "phase0"),
        (learn_arrow_ray, "phase0"),
        (learn_lshape_concavity, "phase0"),
    ]

    # Phase 2 learners
    phase2_learners = [
        (learn_cell_rule, "cell_rule"),
        (learn_region_rule, "region"),
        (learn_context_rule, "context"),
        (learn_ranking_rule, "ranking"),
        (learn_legend_rule, "legend"),
        (learn_compose_rule, "compose"),
        (learn_gravity_rule, "gravity"),
        (learn_tiling_rule, "tiling"),
        (learn_object_rule, "object_rule"),
        (learn_multiobj_rule, "multiobj"),
        (learn_extraction_rule, "extraction"),
        (learn_grid_rule, "grid_ops"),
        (learn_advanced_rule, "advanced"),
        (learn_pattern_rule, "pattern"),
    ]

    results = []
    solver = SmartARCSolverV2()

    for task_data in tasks:
        tid = task_data["id"]
        train = task_data["train"]
        test_input = task_data["test_input"]
        test_output = task_data["test_output"]

        features = extract_feature_vector(train)
        label = "none"

        # Try Phase 0 learners
        for learn_fn, family in phase0_learners:
            try:
                rule = learn_fn(train)
                if rule is not None:
                    result = rule(test_input)
                    if result == test_output:
                        label = family
                        break
            except Exception:
                continue

        # Try Phase 1 (base class) if Phase 0 didn't solve it
        if label == "none":
            try:
                from klomboagi.reasoning.arc_solver import ARCSolverV18
                base = ARCSolverV18()
                result = base.solve(train, test_input)
                if result == test_output:
                    label = "phase1"
            except Exception:
                pass

        # Try Phase 2 learners
        if label == "none":
            for learn_fn, family in phase2_learners:
                try:
                    rule = learn_fn(train)
                    if rule is not None:
                        result = rule(test_input)
                        if result == test_output:
                            label = family
                            break
                except Exception:
                    continue

        # Try DSL
        if label == "none":
            try:
                from klomboagi.reasoning.arc_dsl_v2 import synthesize
                result = synthesize(train, test_input, max_depth=3, timeout_ms=2000)
                if result == test_output:
                    label = "dsl"
            except Exception:
                pass

        results.append({
            "task_id": tid,
            "features": features,
            "label": label,
        })

    return results


def train_classifier(labels_data: list[dict] | None = None, save: bool = True):
    """
    Train a RandomForest classifier on collected label data.

    Returns (classifier, accuracy, classification_report).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    if labels_data is None:
        if LABELS_PATH.exists():
            with open(LABELS_PATH) as f:
                labels_data = json.load(f)
        else:
            print("No labels file found. Running collect_labels()...")
            labels_data = collect_labels()
            with open(LABELS_PATH, "w") as f:
                json.dump(labels_data, f)

    X = np.array([d["features"] for d in labels_data])
    y_raw = [d["label"] for d in labels_data]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Train RandomForest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    # Cross-validation score
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Label distribution
    dist = Counter(y_raw)
    print(f"Label distribution: {dict(dist)}")

    # Fit on all data
    clf.fit(X, y)

    # Feature importance
    fnames = feature_names()
    importances = sorted(zip(fnames, clf.feature_importances_),
                         key=lambda x: -x[1])
    print("Top 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.3f}")

    if save:
        model_data = {"classifier": clf, "label_encoder": le}
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {MODEL_PATH}")

    return clf, le, scores.mean()


def load_classifier():
    """Load trained classifier from disk."""
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["classifier"], data["label_encoder"]


def predict_family(train: list[dict]) -> str | None:
    """
    Predict the best strategy family for a task.
    Returns family name string or None if no model loaded.
    """
    clf, le = load_classifier()
    if clf is None:
        return None

    features = np.array([extract_feature_vector(train)])
    pred = clf.predict(features)[0]
    return le.inverse_transform([pred])[0]


def predict_family_proba(train: list[dict]) -> list[tuple[str, float]] | None:
    """
    Predict strategy family probabilities for a task.
    Returns sorted list of (family, probability) tuples.
    """
    clf, le = load_classifier()
    if clf is None:
        return None

    features = np.array([extract_feature_vector(train)])
    proba = clf.predict_proba(features)[0]
    families = le.inverse_transform(range(len(proba)))
    ranked = sorted(zip(families, proba), key=lambda x: -x[1])
    return ranked


# ── Utility ──

def _load_arckit_tasks() -> list[dict]:
    """Load ARC tasks from arckit in our format."""
    import arckit
    train_set, _ = arckit.load_data()
    tasks = []
    for task in train_set[:400]:
        train = [{"input": [list(int(x) for x in row) for row in inp],
                  "output": [list(int(x) for x in row) for row in out]}
                 for inp, out in task.train]
        test_input = [list(int(x) for x in row) for row in task.test[0][0]]
        test_output = [list(int(x) for x in row) for row in task.test[0][1]]
        tasks.append({
            "id": task.id,
            "train": train,
            "test_input": test_input,
            "test_output": test_output,
        })
    return tasks


if __name__ == "__main__":
    print("Step 1: Collecting labels...")
    labels = collect_labels()

    print(f"\nCollected {len(labels)} labels")
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)
    print(f"Saved to {LABELS_PATH}")

    print("\nStep 2: Training classifier...")
    clf, le, acc = train_classifier(labels)
    print(f"\nDone. Accuracy: {acc:.3f}")
