"""
Self-Reinforcement — the system rewards itself when reasoning works.

From DeepSeek R1: reasoning can emerge from pure reward signals.
No human-labeled reasoning data needed.

From Marcus Hutter (AIXI): the optimal agent maximizes prediction accuracy.
Better predictions = better understanding = smarter.

Combined: the system tracks its own predictions. When a prediction comes true,
the reasoning chain that produced it gets strengthened. When wrong, weakened.

This is how the system IMPROVES without external training.
The reward signal is internal: "did my reasoning match reality?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class Prediction:
    """Something the system predicted would be true."""
    id: int
    statement: str                    # What was predicted
    basis: str                        # What reasoning produced this
    confidence: float                 # How confident at prediction time
    timestamp: float                  # When predicted
    verified: bool = False            # Has it been checked?
    correct: bool | None = None       # Was it right?
    verification_time: float = 0.0    # When checked
    reward_applied: bool = False      # Has the reward been processed?

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "basis": self.basis,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "verified": self.verified,
            "correct": self.correct,
        }


@dataclass
class ReasoningOutcome:
    """Record of how a reasoning chain performed."""
    chain_id: str                     # Which reasoning chain
    approach: str                     # What approach was used
    predicted_outcome: str            # What was expected
    actual_outcome: str               # What actually happened
    success: bool                     # Did prediction match reality?
    reward: float = 0.0               # Calculated reward
    timestamp: float = 0.0


class SelfReinforcement:
    """
    Internal reward system. No external trainer needed.

    The system:
    1. Makes predictions based on its reasoning
    2. Observes what actually happens
    3. Compares prediction to reality
    4. Rewards accurate reasoning chains, penalizes inaccurate ones
    5. Over time, the reasoning strategies that work get strengthened

    This is NOT RL in the traditional sense (no neural network, no gradient).
    It's evidence-based reinforcement through the NARS truth value system:
    - Successful reasoning → revise truth values upward
    - Failed reasoning → revise truth values downward
    """

    def __init__(self) -> None:
        self._predictions: list[Prediction] = []
        self._outcomes: list[ReasoningOutcome] = []
        self._id_counter = 0

        # Strategy performance tracking
        self._strategy_scores: dict[str, list[float]] = {}
        # approach_name → list of rewards (positive or negative)

    def predict(self, statement: str, basis: str, confidence: float) -> Prediction:
        """
        Register a prediction. The system is saying "I think X will be true."
        Later, when we verify, we reward or penalize the reasoning.
        """
        self._id_counter += 1
        pred = Prediction(
            id=self._id_counter,
            statement=statement,
            basis=basis,
            confidence=confidence,
            timestamp=time.time(),
        )
        self._predictions.append(pred)
        return pred

    def verify(self, prediction_id: int, actual_result: bool) -> float:
        """
        Check a prediction against reality.
        Returns the reward (positive = correct, negative = wrong).

        The reward is PROPORTIONAL to confidence:
        - Confident and right → big positive reward
        - Confident and wrong → big negative reward (penalize overconfidence)
        - Uncertain and right → small positive
        - Uncertain and wrong → small negative (no big deal, we weren't sure)
        """
        pred = None
        for p in self._predictions:
            if p.id == prediction_id and not p.verified:
                pred = p
                break

        if pred is None:
            return 0.0

        pred.verified = True
        pred.correct = actual_result
        pred.verification_time = time.time()

        # Calculate reward — Hutter-inspired: proportional to prediction accuracy
        if actual_result:
            # Correct prediction — reward proportional to confidence
            reward = pred.confidence * 0.5  # Max reward 0.5
        else:
            # Wrong prediction — penalty proportional to confidence
            # Being confidently wrong is worse than being uncertainly wrong
            reward = -pred.confidence * 0.3  # Max penalty -0.3

        pred.reward_applied = True

        # Track strategy performance
        if pred.basis not in self._strategy_scores:
            self._strategy_scores[pred.basis] = []
        self._strategy_scores[pred.basis].append(reward)

        return reward

    def observe_outcome(self, chain_id: str, approach: str,
                        predicted: str, actual: str,
                        success: bool) -> ReasoningOutcome:
        """
        Record the outcome of a reasoning chain.
        This is the broader version — not just true/false predictions,
        but how well the overall reasoning worked.
        """
        reward = 0.2 if success else -0.1
        outcome = ReasoningOutcome(
            chain_id=chain_id,
            approach=approach,
            predicted_outcome=predicted,
            actual_outcome=actual,
            success=success,
            reward=reward,
            timestamp=time.time(),
        )
        self._outcomes.append(outcome)

        # Track strategy
        if approach not in self._strategy_scores:
            self._strategy_scores[approach] = []
        self._strategy_scores[approach].append(reward)

        return outcome

    def best_strategy(self) -> str | None:
        """What reasoning approach has worked best?"""
        if not self._strategy_scores:
            return None

        avg_scores = {}
        for strategy, scores in self._strategy_scores.items():
            if scores:
                avg_scores[strategy] = sum(scores) / len(scores)

        if not avg_scores:
            return None

        return max(avg_scores, key=lambda s: avg_scores[s])

    def strategy_ranking(self) -> list[tuple[str, float, int]]:
        """Rank all strategies by average reward."""
        rankings = []
        for strategy, scores in self._strategy_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                rankings.append((strategy, avg, len(scores)))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def should_explore(self, current_strategy: str) -> bool:
        """
        Should the system try a different approach?

        If the current strategy has been failing, YES — explore.
        If it's been working, NO — exploit.

        This is the explore/exploit tradeoff from RL,
        but based on evidence counts, not epsilon-greedy.
        """
        scores = self._strategy_scores.get(current_strategy, [])
        if len(scores) < 3:
            return False  # Not enough data to judge

        recent = scores[-5:]  # Last 5 uses
        avg_recent = sum(recent) / len(recent)

        return avg_recent < 0.0  # Explore if recent performance is negative

    def prediction_accuracy(self) -> float:
        """Overall prediction accuracy."""
        verified = [p for p in self._predictions if p.verified]
        if not verified:
            return 0.0
        correct = sum(1 for p in verified if p.correct)
        return correct / len(verified)

    def pending_predictions(self) -> list[Prediction]:
        """Predictions waiting to be verified."""
        return [p for p in self._predictions if not p.verified]

    def stats(self) -> dict:
        verified = [p for p in self._predictions if p.verified]
        correct = [p for p in verified if p.correct]
        return {
            "total_predictions": len(self._predictions),
            "verified": len(verified),
            "correct": len(correct),
            "accuracy": self.prediction_accuracy(),
            "strategies_tried": len(self._strategy_scores),
            "best_strategy": self.best_strategy(),
            "total_outcomes": len(self._outcomes),
        }
