"""Tests for the memory model, the graph, the schedulers and the study's claims."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.baselines import Fixed, Leitner, SM2  # noqa: E402
from model.hlr import (  # noqa: E402
    MAX_HALF_LIFE, MIN_HALF_LIFE, auc, fit, implied_half_life,
    mean_absolute_error, recall,
)
from model.simulate import (  # noqa: E402
    FEATURES, feature_matrix, load_concepts, simulate, split_by_student, topological,
)
from schedule.planner import HalfLifePlanner, PrerequisiteFirstPlanner, reviews_needed, run_schedule  # noqa: E402


# --- the forgetting curve ---------------------------------------------------

def test_recall_is_one_half_after_one_half_life():
    assert recall(1.0, 1.0) == pytest.approx(0.5)
    assert recall(2.0, 1.0) == pytest.approx(0.25)
    assert recall(0.0, 3.0) == pytest.approx(1.0)


def test_recall_decays_with_time_and_rises_with_half_life():
    assert recall(5.0, 2.0) < recall(1.0, 2.0)
    assert recall(5.0, 10.0) > recall(5.0, 2.0)


def test_the_implied_half_life_is_bounded_at_both_ends():
    """A single binary answer implies an unbounded or zero half-life.

    Two guards act here and they bind in different places. Clipping the observed
    probability away from 0 and 1 is what actually tames the extremes; the
    half-life clamp is the outer backstop and usually does not bind. A correct
    answer saturates at the ceiling, a wrong one lands near the floor but above
    it, and a genuine one-half observation is exact.
    """
    assert implied_half_life(1.0, 1.0) == pytest.approx(MAX_HALF_LIFE)
    assert MIN_HALF_LIFE <= implied_half_life(1.0, 0.0) < 0.1
    assert implied_half_life(2.0, 0.5) == pytest.approx(2.0)
    assert implied_half_life(100.0, 1.0) == pytest.approx(MAX_HALF_LIFE), "the ceiling binds"
    # The degeneracy the alpha=0 default exists for: both binary outcomes land
    # at an extreme, so the term carries almost no information about one review.
    assert implied_half_life(1.0, 1.0) / max(implied_half_life(1.0, 0.0), 1e-9) > 1000


def test_auc_of_a_constant_predictor_is_one_half():
    assert auc(np.ones(10), np.array([0, 1] * 5)) == pytest.approx(0.5)
    assert auc(np.arange(6), np.array([0, 0, 0, 1, 1, 1])) == pytest.approx(1.0)
    assert np.isnan(auc(np.arange(4), np.zeros(4)))


# --- the concept graph ------------------------------------------------------

def test_the_course_graph_is_acyclic_and_ordered():
    concepts, prereqs, names = load_concepts()
    assert len(concepts) > 40
    assert set(concepts) == set(prereqs)
    position = {c: i for i, c in enumerate(concepts)}
    for concept, parents in prereqs.items():
        for parent in parents:
            assert position[parent] < position[concept], f"{parent} must come before {concept}"
    assert "chain-rule" in prereqs["implicit-differentiation"], "the worked example in the README"


def test_a_cycle_is_refused():
    with pytest.raises(ValueError, match="cycle"):
        topological({"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError, match="unknown concept"):
        topological({"a": ["ghost"]})


# --- fitting ----------------------------------------------------------------

@pytest.fixture(scope="module")
def coupled_data():
    concepts, prereqs, _ = load_concepts()
    sim = simulate(gamma_true=0.8, students=30, seed=99, concepts=concepts, prereqs=prereqs)
    return sim, split_by_student(sim)


def test_simulation_produces_usable_histories(coupled_data):
    sim, (train, test) = coupled_data
    assert len(train) > 2000 and len(test) > 500
    assert {r.student for r in train}.isdisjoint({r.student for r in test}), "split on students"
    accuracy = np.mean([r.correct for r in sim.reviews])
    assert 0.2 < accuracy < 0.9, "a simulation nobody ever fails teaches nothing"


def test_gamma_is_zero_when_the_data_has_no_coupling():
    """The null case. The extension must not invent a coupling."""
    concepts, prereqs, _ = load_concepts()
    sim = simulate(gamma_true=0.0, students=30, seed=12, concepts=concepts, prereqs=prereqs)
    train, _ = split_by_student(sim)
    model = fit(feature_matrix(train), np.array([r.delta for r in train]),
                np.array([r.correct for r in train], dtype=float),
                np.array([r.prereq_recall for r in train]), couple=True, feature_names=FEATURES)
    assert abs(model.gamma) < 0.2, f"found a coupling that is not there: {model.gamma}"


def test_gamma_is_recovered_when_the_coupling_is_real(coupled_data):
    _, (train, _) = coupled_data
    model = fit(feature_matrix(train), np.array([r.delta for r in train]),
                np.array([r.correct for r in train], dtype=float),
                np.array([r.prereq_recall for r in train]), couple=True, feature_names=FEATURES)
    # Attenuated, because the prerequisite recall fed to the fit is itself an
    # estimate. Monotone and clearly positive is the claim, not exactness.
    assert 0.3 < model.gamma < 1.2, f"gamma {model.gamma} does not track a true 0.8"


def test_coupling_improves_held_out_prediction(coupled_data):
    _, (train, test) = coupled_data
    x_train, delta_train = feature_matrix(train), np.array([r.delta for r in train])
    y_train = np.array([r.correct for r in train], dtype=float)
    prereq_train = np.array([r.prereq_recall for r in train])
    base = fit(x_train, delta_train, y_train, feature_names=FEATURES)
    coupled = fit(x_train, delta_train, y_train, prereq_train, couple=True, feature_names=FEATURES)

    x_test, delta_test = feature_matrix(test), np.array([r.delta for r in test])
    y_test = np.array([r.correct for r in test], dtype=float)
    prereq_test = np.array([r.prereq_recall for r in test])
    base_mae = mean_absolute_error(base.predict(x_test, delta_test), y_test)
    coupled_mae = mean_absolute_error(coupled.predict(x_test, delta_test, prereq_test), y_test)
    assert coupled_mae < base_mae


def test_gamma_zero_reproduces_the_base_model_exactly():
    """The extension is nested inside what it is compared against."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(40, len(FEATURES)))
    delta = rng.uniform(0.1, 10, size=40)
    model = fit(features, delta, rng.random(40), feature_names=FEATURES)
    assert model.gamma == 0.0
    with_prereq = model.predict(features, delta, rng.random(40))
    without = model.predict(features, delta)
    assert np.allclose(with_prereq, without)


# --- schedulers -------------------------------------------------------------

def test_baselines_behave_as_published():
    sm2 = SM2()
    assert sm2.next_interval("a", True) == pytest.approx(1.0)
    assert sm2.next_interval("a", True) == pytest.approx(6.0)
    assert sm2.next_interval("a", True) > 6.0, "the third correct answer stretches the interval"
    assert sm2.next_interval("a", False) == pytest.approx(1.0), "a wrong answer resets"
    assert sm2.ease["a"] >= 1.3, "ease never falls below the published floor"

    leitner = Leitner()
    assert leitner.next_interval("a", True) == 2.0
    assert leitner.next_interval("a", False) == 1.0
    assert Fixed(days=4.0).next_interval("a", False) == 4.0


def test_the_planner_does_not_chase_concepts_that_are_already_lost():
    """Reviewing below one half is expected to shorten the half-life, not lengthen it."""
    prereqs = {"a": [], "b": [], "c": []}
    planner = HalfLifePlanner(half_lives={"a": 1.0, "b": 1.0, "c": 1.0}, prereqs=prereqs)
    # a is long gone, b is about to slip, c was just reviewed.
    last_seen = {"a": 0.0, "b": 9.0, "c": 9.9}
    assert planner.choose(10.0, ["a", "b", "c"], last_seen) == "b"


def test_the_planner_falls_back_to_the_best_chance_when_everything_has_slipped():
    prereqs = {"a": [], "b": []}
    planner = HalfLifePlanner(half_lives={"a": 1.0, "b": 1.0}, prereqs=prereqs)
    assert planner.choose(10.0, ["a", "b"], {"a": 0.0, "b": 7.0}) == "b"


def test_a_planner_with_gamma_zero_matches_the_plain_planner():
    concepts, prereqs, _ = load_concepts()
    rng = np.random.default_rng(5)
    half_lives = {c: float(rng.lognormal(np.log(2.0), 0.4)) for c in concepts}
    plain = run_schedule(HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs, gamma=0.0),
                         half_lives, prereqs, concepts, gamma_true=0.0, seed=1)
    coupled = run_schedule(HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs, gamma=0.0),
                           half_lives, prereqs, concepts, gamma_true=0.0, seed=1)
    assert plain["mean_recall"] == pytest.approx(coupled["mean_recall"])


def test_more_reviews_never_hurt():
    concepts, prereqs, _ = load_concepts()
    rng = np.random.default_rng(6)
    half_lives = {c: float(rng.lognormal(np.log(2.0), 0.4)) for c in concepts}
    low = run_schedule(HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs),
                       half_lives, prereqs, concepts, budget_per_day=5, seed=2)
    high = run_schedule(HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs),
                        half_lives, prereqs, concepts, budget_per_day=25, seed=2)
    assert high["mean_recall"] > low["mean_recall"]
    assert high["reviews"] > low["reviews"]


def test_the_budget_search_returns_a_budget_that_actually_holds():
    concepts, prereqs, _ = load_concepts()
    rng = np.random.default_rng(7)
    half_lives = {c: float(rng.lognormal(np.log(2.0), 0.4)) for c in concepts}
    found = reviews_needed(lambda: HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs),
                           half_lives, prereqs, concepts, gamma_true=0.0, seed=3, hi=60)
    assert found["budget"] is not None
    assert found["recall_at_budget"] >= 0.8
    below = run_schedule(HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs),
                         half_lives, prereqs, concepts, budget_per_day=found["budget"] - 1,
                         gamma_true=0.0, seed=3)
    assert below["mean_recall"] < 0.8, "the search must return the smallest budget that works"


def test_coupling_helps_scheduling_only_under_scarcity():
    """The project's main negative-then-positive result, pinned.

    With a budget big enough to hold everything above target, a scheduler keeps
    prerequisites fresh anyway, so the coupling term stays near one and knowing
    about it changes nothing. Starve the budget and it matters a great deal.
    """
    concepts, prereqs, _ = load_concepts()
    rng = np.random.default_rng(8)
    half_lives = {c: float(rng.lognormal(np.log(2.0), 0.4)) for c in concepts}

    def held(gamma, budget):
        return run_schedule(
            HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs, gamma=gamma),
            half_lives, prereqs, concepts, budget_per_day=budget, gamma_true=0.8, seed=4,
        )["mean_recall"]

    assert held(0.8, 15) > held(0.0, 15) * 1.3, "under scarcity, coupling should help clearly"


def test_prerequisite_first_is_not_an_improvement():
    """A negative result worth keeping: the intuitive fix is worse."""
    concepts, prereqs, _ = load_concepts()
    rng = np.random.default_rng(9)
    half_lives = {c: float(rng.lognormal(np.log(2.0), 0.4)) for c in concepts}
    coupled = run_schedule(HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs, gamma=0.8),
                           half_lives, prereqs, concepts, budget_per_day=15, gamma_true=0.8, seed=5)
    prereq_first = run_schedule(
        PrerequisiteFirstPlanner(half_lives=dict(half_lives), prereqs=prereqs, gamma=0.8),
        half_lives, prereqs, concepts, budget_per_day=15, gamma_true=0.8, seed=5)
    assert prereq_first["mean_recall"] < coupled["mean_recall"]
