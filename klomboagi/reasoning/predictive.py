"""
Predictive Processing — the brain as a prediction machine.

From Friston, Clark, Hohwy — Predictive Precision Weighting.

The core idea: the brain doesn't passively receive input.
It PREDICTS what will happen next, then compares prediction to reality.
The difference (prediction error) is the learning signal.

High precision = trust the prediction error (learn from it).
Low precision = ignore the prediction error (noise, not signal).

For KlomboAGI:
  Before answering a question, PREDICT what the answer should be.
  After getting feedback, compare prediction to reality.
  Large prediction error + high precision = strong learning signal.
  Large prediction error + low precision = ignore (noise).

This replaces uniform learning with WEIGHTED learning —
the system learns MORE from surprising, reliable signals
and LESS from noisy, expected ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Prediction:
    """A prediction the system made before seeing the outcome."""
    concept: str
    predicted: str          # what we expected
    confidence: float       # how confident in the prediction
    precision: float = 0.5  # how reliable this domain is (learned)

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "predicted": self.predicted[:60],
            "confidence": round(self.confidence, 3),
            "precision": round(self.precision, 3),
        }


@dataclass
class PredictionError:
    """The difference between prediction and reality."""
    prediction: Prediction
    actual: str
    error_magnitude: float   # 0-1, how wrong we were
    weighted_error: float    # error * precision (the real learning signal)
    should_learn: bool       # is this worth updating beliefs for?

    def to_dict(self) -> dict:
        return {
            "predicted": self.prediction.predicted[:40],
            "actual": self.actual[:40],
            "error": round(self.error_magnitude, 3),
            "weighted": round(self.weighted_error, 3),
            "learn": self.should_learn,
        }


class PredictiveProcessor:
    """
    Predict → Compare → Learn (weighted by precision).

    Tracks prediction accuracy per domain. Domains where the system
    is consistently right get HIGH precision (trust errors there).
    Domains where it's consistently wrong get LOW precision
    (errors there are expected, not informative).
    """

    LEARNING_THRESHOLD = 0.2  # Only learn from errors above this

    def __init__(self) -> None:
        self._domain_precision: dict[str, float] = {}  # domain → precision
        self._prediction_history: list[PredictionError] = []

    def predict(self, concept: str, beliefs: dict) -> Prediction:
        """
        Before seeing the answer, predict what it should be.
        Uses existing beliefs to form expectation.
        """
        # Find relevant beliefs
        predicted = ""
        confidence = 0.0
        for stmt, belief in beliefs.items():
            if hasattr(belief, 'subject') and belief.subject == concept.lower():
                if belief.predicate and len(belief.predicate) < 80:
                    predicted = belief.predicate
                    confidence = belief.truth.confidence if hasattr(belief, 'truth') else 0.5
                    break

        precision = self._domain_precision.get(concept.lower(), 0.5)

        return Prediction(
            concept=concept,
            predicted=predicted or "unknown",
            confidence=confidence,
            precision=precision,
        )

    def compare(self, prediction: Prediction, actual: str) -> PredictionError:
        """
        Compare prediction to reality. Compute weighted error.
        """
        # How different are prediction and actual?
        if not prediction.predicted or prediction.predicted == "unknown":
            error = 1.0  # No prediction at all
        elif prediction.predicted.lower() in actual.lower():
            error = 0.0  # Exact substring match
        else:
            # Check if we got an answer about the RIGHT CONCEPT
            # (even if the words differ, did we answer about the same thing?)
            concept = prediction.concept.lower()
            actual_lower = actual.lower()

            # If the concept appears in the answer AND the answer isn't "don't know"
            if concept in actual_lower and "don't know" not in actual_lower:
                # We answered about the right thing — partial success
                # Check word overlap for precision
                stop = {"is", "a", "an", "the", "of", "and", "or", "in", "to", "for",
                        "with", "that", "this", "by", "from", "on", "at", "as", "it",
                        "causes", "enables", "part", "contains", "means", "uses"}
                pred_words = set(prediction.predicted.lower().split()) - stop
                actual_words = set(actual_lower.split()) - stop

                if pred_words:
                    overlap = len(pred_words & actual_words) / len(pred_words)
                    error = max(0.0, 0.5 - overlap)  # Base error 0.5 (right concept), reduced by overlap
                else:
                    error = 0.3  # Right concept, no words to compare
            elif "don't know" in actual_lower:
                error = 1.0  # Complete miss
            else:
                # Wrong concept or no concept match
                stop = {"is", "a", "an", "the", "of", "and", "or", "in", "to", "for",
                        "with", "that", "this", "by", "from", "on", "at", "as", "it"}
                pred_words = set(prediction.predicted.lower().split()) - stop
                actual_words = set(actual_lower.split()) - stop
                if pred_words and actual_words:
                    overlap = len(pred_words & actual_words) / len(pred_words)
                    error = 1.0 - overlap
                else:
                    error = 1.0

        # Weighted error = error * precision
        weighted = error * prediction.precision

        # Should we learn from this?
        should_learn = weighted > self.LEARNING_THRESHOLD

        pe = PredictionError(
            prediction=prediction,
            actual=actual,
            error_magnitude=error,
            weighted_error=weighted,
            should_learn=should_learn,
        )
        self._prediction_history.append(pe)
        if len(self._prediction_history) > 100:
            self._prediction_history = self._prediction_history[-100:]

        # Update domain precision based on accuracy
        self._update_precision(prediction.concept, error)

        return pe

    def _update_precision(self, concept: str, error: float) -> None:
        """
        Update precision for a domain based on prediction accuracy.

        Consistently accurate → precision rises (trust errors more).
        Consistently inaccurate → precision falls (errors are noise).
        """
        key = concept.lower()
        current = self._domain_precision.get(key, 0.5)

        if error < 0.3:
            # We predicted correctly → precision increases
            new = min(0.95, current + 0.05)
        else:
            # We predicted wrong → precision decreases
            new = max(0.05, current - 0.03)

        self._domain_precision[key] = new

    def accuracy(self) -> float:
        """Overall prediction accuracy. Correct = error < 0.5 (right concept + some overlap)."""
        if not self._prediction_history:
            return 0.0
        correct = sum(1 for pe in self._prediction_history
                     if pe.error_magnitude < 0.5 and pe.prediction.predicted != "unknown")
        # Only count predictions where we actually predicted something
        predicted = sum(1 for pe in self._prediction_history
                       if pe.prediction.predicted != "unknown")
        return correct / max(1, predicted)

    def stats(self) -> dict:
        return {
            "predictions_made": len(self._prediction_history),
            "accuracy": round(self.accuracy(), 3),
            "high_precision_domains": [
                k for k, v in self._domain_precision.items() if v > 0.7
            ][:10],
        }
