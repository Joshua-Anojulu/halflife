"""Scheduling, and the 30-day comparison that answers the question a student has.

A scheduler's job is to pick what to review next. The half-life planner picks the
concept whose effective recall has fallen furthest below the target, walking the
dependency graph so a prerequisite is never scheduled after the thing that
depends on it.

The comparison gives every scheduler the same simulated student, the same
starting memory, the same review budget per day, and the same forgetting
process. Nothing about the simulated student is visible to any scheduler: they
all see only their own history of right and wrong answers, which is what a real
scheduler has.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.hlr import MAX_HALF_LIFE, MIN_HALF_LIFE, clamp, recall, recall_scalar
from model.simulate import topological

TARGET = 0.5   # conventional retrievability target for scheduling
HOLD = 0.8     # the level the comparison asks each scheduler to hold


@dataclass
class PrerequisiteFirstPlanner:
    """Fixes what the coupled planner gets wrong.

    Coupling tells you a dependent concept is decaying because its prerequisite
    has. Scheduling the *dependent* more often is the wrong response: the
    prerequisite is what decayed, and reviewing it lifts every concept that
    rests on it at once. Since h_eff(i) = h(i) * prod p(j) ** gamma, restoring
    one prerequisite multiplies the effective half-life of each of its
    dependents, so a prerequisite with many decayed dependents is worth more
    than any one of them.

    Score a candidate by its own urgency plus the number of its dependents that
    have slipped below target, weighted by gamma. gamma = 0 makes this the
    plain planner exactly.
    """
    half_lives: dict
    prereqs: dict
    gamma: float = 0.0
    target: float = TARGET
    name: str = "Half-life, prerequisite-first"
    _dependents: dict | None = None
    _cached_order: list | None = None

    def dependents(self) -> dict:
        if self._dependents is None:
            built: dict = {}
            for concept, parents in self.prereqs.items():
                for parent in parents:
                    built.setdefault(parent, []).append(concept)
            self._dependents = built
        return self._dependents

    def predicted_recall(self, concept: str, now: float, last_seen: dict) -> float:
        if concept not in last_seen:
            return 0.0
        return recall_scalar(max(now - last_seen[concept], 0.0), clamp(self.half_lives.get(concept, 1.0)))

    def choose(self, now: float, available, last_seen: dict) -> str | None:
        if not available:
            return None
        recalls = {c: self.predicted_recall(c, now, last_seen) for c in available}
        at_risk = [c for c in available if recalls[c] >= self.target]
        pool = at_risk or available
        best, best_score = None, -1e9
        for concept in pool:
            slipped = sum(1 for child in self.dependents().get(concept, ())
                          if recalls.get(child, 0.0) < self.target)
            # Lower own recall is more urgent; leverage over dependents adds to it.
            score = (1.0 - recalls[concept]) + self.gamma * slipped
            if not at_risk:
                score = recalls[concept] + self.gamma * slipped
            if score > best_score:
                best, best_score = concept, score
        return best


@dataclass
class HalfLifePlanner:
    """Schedules by predicted effective recall, optionally coupling prerequisites."""
    half_lives: dict
    prereqs: dict
    gamma: float = 0.0
    target: float = TARGET
    name: str = "Half-life"
    _cached_order: list | None = None

    def effective(self, concept: str, now: float, last_seen: dict) -> float:
        base = self.half_lives.get(concept, 1.0)
        if self.gamma:
            coupling = 1.0
            for parent in self.prereqs.get(concept, ()):
                coupling *= self.predicted_recall(parent, now, last_seen)
            base *= max(coupling, 1e-4) ** self.gamma
        return clamp(base)

    def predicted_recall(self, concept: str, now: float, last_seen: dict) -> float:
        if concept not in last_seen:
            return 0.0
        return recall_scalar(max(now - last_seen[concept], 0.0), clamp(self.half_lives.get(concept, 1.0)))

    def choose(self, now: float, available, last_seen: dict) -> str | None:
        """Which concept to review now.

        Not "whichever is most forgotten". Under this memory model a review
        multiplies the half-life by g when answered correctly and divides by g
        when answered wrongly, so the expected change in log half-life is
        (2p - 1) * log(g). Reviewing an item whose recall has already fallen
        below one half is expected to make it *worse*, because the wrong answer
        is the likely one. Greedily chasing the most-forgotten item is therefore
        the wrong policy, and it is what a naive scheduler does.

        So: among items at or above the target, take the lowest, the one closest
        to slipping. If every item has already slipped, take the highest, which
        is the one most likely to come back with a correct answer.
        """
        if not available:
            return None
        order = {c: i for i, c in enumerate(self._order(available))}
        at_risk, lost = [], []
        for concept in available:
            if concept not in last_seen:
                at_risk.append((-10.0, order.get(concept, 0), concept))
                continue
            h = self.effective(concept, now, last_seen)
            p = recall_scalar(max(now - last_seen[concept], 0.0), h)
            (at_risk if p >= self.target else lost).append((p, order.get(concept, 0), concept))
        if at_risk:
            at_risk.sort()
            return at_risk[0][2]
        lost.sort(reverse=True)
        return lost[0][2]

    def _order(self, available):
        if self._cached_order is None:
            self._cached_order = topological({k: self.prereqs.get(k, []) for k in available})
        return self._cached_order


def run_schedule(scheduler, student_half_lives, prereqs, concepts, *, days=30.0,
                 budget_per_day=10, gamma_true=0.0, growth=2.0, seed=0) -> dict:
    """One simulated month. Returns mean recall held and reviews spent.

    The student forgets according to gamma_true, which no scheduler knows.
    """
    rng = np.random.default_rng(seed)
    half_lives = dict(student_half_lives)
    last_seen = {c: 0.0 for c in concepts}
    reviews = 0
    samples = []

    def true_recall(concept: str, now: float) -> float:
        coupling = 1.0
        if gamma_true:
            for parent in prereqs.get(concept, ()):
                coupling *= recall_scalar(max(now - last_seen[parent], 0.0), clamp(half_lives[parent]))
            coupling = max(coupling, 1e-4) ** gamma_true
        effective = clamp(half_lives[concept] * coupling)
        return recall_scalar(max(now - last_seen[concept], 0.0), effective)

    for day in range(int(days)):
        # Reviews are spread through the day. Without this, every review in a
        # session sees delta = 0 for the item just answered, so a recall-based
        # policy keeps picking the same concept and the day is wasted.
        for slot in range(budget_per_day):
            now = day + (slot / max(budget_per_day, 1))
            if hasattr(scheduler, "choose"):
                concept = scheduler.choose(now, concepts, last_seen)
            else:
                due = [(scheduler.due_at(c, last_seen[c]), c) for c in concepts]
                due.sort()
                concept = due[0][1]
            if concept is None:
                break
            probability = true_recall(concept, now)
            correct = bool(rng.random() < probability)
            reviews += 1
            factor = growth if correct else 1.0 / growth
            half_lives[concept] = clamp(half_lives[concept] * factor)
            last_seen[concept] = now
            if hasattr(scheduler, "next_interval"):
                scheduler.next_interval(concept, correct)
            if hasattr(scheduler, "half_lives"):
                scheduler.half_lives[concept] = half_lives[concept]
        # Measure at the end of the day, before the next day's reviews.
        samples.append(float(np.mean([true_recall(c, day + 1.0) for c in concepts])))

    held = float(np.mean(samples))
    return {
        "scheduler": getattr(scheduler, "name", type(scheduler).__name__),
        "mean_recall": held,
        "final_recall": samples[-1],
        "reviews": reviews,
        "days_above_hold": int(sum(1 for s in samples if s >= HOLD)),
        "daily": samples,
    }


def reviews_needed(make_scheduler, student_half_lives, prereqs, concepts, *, days=30.0,
                   gamma_true=0.0, hold=HOLD, seed=0, lo=1, hi=60) -> dict:
    """Smallest daily review budget that holds mean recall at or above `hold`.

    This is the number a student cares about: not "how good is recall at some
    arbitrary budget" but "how much work does this schedule cost me to stay on
    top of the material". Monotone in the budget, so a binary search is exact
    to one review per day. Returns None when even the ceiling fails.
    """
    def held(budget: int) -> float:
        return run_schedule(make_scheduler(), student_half_lives, prereqs, concepts,
                            days=days, budget_per_day=budget, gamma_true=gamma_true,
                            seed=seed)["mean_recall"]

    if held(hi) < hold:
        return {"budget": None, "ceiling": hi, "recall_at_ceiling": held(hi)}
    low, high = lo, hi
    while low < high:
        mid = (low + high) // 2
        if held(mid) >= hold:
            high = mid
        else:
            low = mid + 1
    return {"budget": low, "recall_at_budget": held(low)}
