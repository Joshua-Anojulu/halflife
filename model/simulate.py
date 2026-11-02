"""Synthetic students with a known decay model, so there is ground truth.

The point of simulating is not to show the model works. It is to answer a
question that real data cannot: when the truth has a known coupling strength,
does the fitted model find it, and when the truth has none, does the model
correctly report none?

That is parameter recovery, and it is the only honest way to evaluate an
extension on data you generated yourself. A study that simulates coupling and
then reports that the coupled model wins has shown nothing except that the
code runs.

The generative model
--------------------
Each concept has a base half-life drawn per student. Practising a concept
multiplies its half-life by a growth factor on a correct answer and shrinks it
on an incorrect one. Recall at review time is 2 ** (-delta / h_eff), where

    h_eff(i) = h(i) * prod over prereqs j of p(j) ** gamma_true

The answer is then a Bernoulli draw on that recall probability. gamma_true is an
input to the simulation, swept by study.py, and never read by any model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hlr import MAX_HALF_LIFE, MIN_HALF_LIFE, recall

DATA = Path(__file__).resolve().parent.parent / "data"


def load_concepts(path: str | Path | None = None) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    source = Path(path) if path else DATA / "concepts" / "calc3.json"
    payload = json.loads(source.read_text())
    prereqs = {c["id"]: list(c["prereqs"]) for c in payload["concepts"]}
    names = {c["id"]: c["name"] for c in payload["concepts"]}
    order = topological(prereqs)
    return order, prereqs, names


def topological(prereqs: dict[str, list[str]]) -> list[str]:
    """Concept ids with every prerequisite before its dependents. Raises on a cycle."""
    visited: dict[str, int] = {}
    order: list[str] = []

    def visit(node: str, trail: tuple[str, ...] = ()) -> None:
        state = visited.get(node, 0)
        if state == 1:
            raise ValueError(f"Prerequisite cycle: {' -> '.join(trail + (node,))}")
        if state == 2:
            return
        visited[node] = 1
        for parent in prereqs.get(node, ()):
            if parent not in prereqs:
                raise ValueError(f"{node} requires unknown concept {parent}")
            visit(parent, trail + (node,))
        visited[node] = 2
        order.append(node)

    for node in prereqs:
        visit(node)
    return order


@dataclass
class Review:
    student: int
    concept: str
    day: float
    delta: float          # days since this concept was last practised
    correct: int
    seen: int             # practices of this concept before now
    right: int
    wrong: int
    prereq_recall: float  # product of prerequisite recall at review time, 1.0 if none
    true_recall: float    # the probability the draw came from; never shown to a model


FEATURES = ("bias", "sqrt_right", "sqrt_wrong", "log_seen", "log_days_known")


def features_for(review: Review) -> list[float]:
    return [
        1.0,
        np.sqrt(review.right),
        np.sqrt(review.wrong),
        np.log1p(review.seen),
        np.log1p(max(review.day, 0.0)),
    ]


def feature_matrix(reviews) -> np.ndarray:
    return np.array([features_for(r) for r in reviews], dtype=float)


@dataclass
class Simulation:
    reviews: list[Review]
    gamma_true: float
    students: int
    concepts: list[str]
    meta: dict = field(default_factory=dict)


def simulate(gamma_true: float = 0.8, students: int = 60, days: float = 60.0,
             reviews_per_day: float = 12.0, seed: int = 20261121,
             concepts=None, prereqs=None) -> Simulation:
    """Practice histories from a known decay model with a known coupling."""
    if concepts is None or prereqs is None:
        concepts, prereqs, _ = load_concepts()
    rng = np.random.default_rng(seed)
    reviews: list[Review] = []

    for student in range(students):
        # Per-student memory strength, so the fit cannot rely on one global scale.
        strength = float(rng.lognormal(mean=np.log(2.0), sigma=0.5))
        growth = float(rng.uniform(1.6, 2.6))
        half_life = {c: strength * float(rng.lognormal(0.0, 0.4)) for c in concepts}
        last_seen: dict[str, float] = {}
        right = {c: 0 for c in concepts}
        wrong = {c: 0 for c in concepts}
        seen = {c: 0 for c in concepts}

        # A student meets concepts roughly in course order, not at random.
        schedule = sorted(
            ((i / len(concepts)) * days * 0.7 + float(rng.uniform(0, days * 0.1)), c)
            for i, c in enumerate(concepts)
        )
        introduced: dict[str, float] = {}
        count = int(rng.poisson(reviews_per_day * days))
        times = np.sort(rng.uniform(0, days, size=count))

        for day in times:
            for when, concept in schedule:
                if when <= day and concept not in introduced:
                    introduced[concept] = when
            available = [c for c in introduced if c in last_seen or introduced[c] <= day]
            if not available:
                continue
            # Prefer concepts not seen for a while, which is what any scheduler does.
            gaps = np.array([day - last_seen.get(c, introduced[c]) for c in available])
            weights = np.maximum(gaps, 0.01)
            concept = available[int(rng.choice(len(available), p=weights / weights.sum()))]

            previous = last_seen.get(concept, introduced[concept])
            delta = max(day - previous, 0.0)
            base = np.clip(half_life[concept], MIN_HALF_LIFE, MAX_HALF_LIFE)

            prereq_p = 1.0
            for parent in prereqs.get(concept, ()):
                if parent in last_seen:
                    parent_delta = max(day - last_seen[parent], 0.0)
                    prereq_p *= float(recall(parent_delta, half_life[parent]))
                elif parent in introduced:
                    prereq_p *= float(recall(max(day - introduced[parent], 0.0), half_life[parent]))
            prereq_p = float(np.clip(prereq_p, 1e-4, 1.0))

            effective = float(np.clip(base * prereq_p ** gamma_true, MIN_HALF_LIFE, MAX_HALF_LIFE))
            probability = float(recall(delta, effective))
            correct = int(rng.random() < probability)

            reviews.append(Review(
                student=student, concept=concept, day=float(day), delta=float(delta),
                correct=correct, seen=seen[concept], right=right[concept], wrong=wrong[concept],
                prereq_recall=prereq_p, true_recall=probability,
            ))

            seen[concept] += 1
            if correct:
                right[concept] += 1
                half_life[concept] = float(np.clip(base * growth, MIN_HALF_LIFE, MAX_HALF_LIFE))
            else:
                wrong[concept] += 1
                half_life[concept] = float(np.clip(base / growth, MIN_HALF_LIFE, MAX_HALF_LIFE))
            last_seen[concept] = float(day)

    return Simulation(reviews=reviews, gamma_true=gamma_true, students=students,
                      concepts=list(concepts),
                      meta={"days": days, "reviews_per_day": reviews_per_day, "seed": seed})


def split_by_student(simulation: Simulation, train_fraction: float = 0.7):
    """Split on students, never on reviews.

    Splitting on reviews would put the same student's earlier and later answers
    on both sides, and the model would score well by learning that student.
    """
    cut = int(simulation.students * train_fraction)
    train = [r for r in simulation.reviews if r.student < cut]
    test = [r for r in simulation.reviews if r.student >= cut]
    return train, test
