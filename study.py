"""The study: can the model recover a coupling strength it was not told?

Simulating students with prerequisite coupling and then reporting that a
coupled model beats an uncoupled one would prove nothing. So the experiment is
a sweep over the true coupling strength, including zero:

1. Parameter recovery. For each gamma_true, simulate fresh students, fit the
   coupled model, and compare the fitted gamma with the one used to generate
   the data. A model that works tracks the truth. A model that always reports
   a positive coupling is finding what it was built to find.

2. The null case. At gamma_true = 0 the coupled model must not beat the base
   model on held-out students. If it does, the extra parameter is fitting
   noise, and that is reported rather than buried.

3. Held-out accuracy. MAE on predicted recall and AUC for predicting a correct
   answer, on students the model never saw.

4. The 30-day schedule comparison, against SM-2, Leitner and fixed intervals,
   which is the number a student actually cares about.

Run: python study.py [--students N] [--trials N] [--quick]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from model.baselines import Fixed, Leitner, SM2
from model.hlr import auc, fit, mean_absolute_error
from model.simulate import (
    FEATURES, feature_matrix, load_concepts, simulate, split_by_student,
)
from schedule.planner import HOLD, HalfLifePlanner, PrerequisiteFirstPlanner, reviews_needed, run_schedule

GAMMAS = (0.0, 0.4, 0.8, 1.2)


def evaluate(reviews, model, use_coupling: bool) -> dict:
    features = feature_matrix(reviews)
    delta = np.array([r.delta for r in reviews])
    observed = np.array([r.correct for r in reviews], dtype=float)
    prereq = np.array([r.prereq_recall for r in reviews]) if use_coupling else None
    predicted = model.predict(features, delta, prereq)
    return {
        "mae": mean_absolute_error(predicted, observed),
        "auc": auc(predicted, observed),
        "n": len(reviews),
    }


def recovery(students: int, trials: int, seed0: int = 20261121) -> list[dict]:
    concepts, prereqs, _ = load_concepts()
    rows = []
    for gamma_true in GAMMAS:
        for trial in range(trials):
            seed = seed0 + trial * 101 + int(gamma_true * 1000)
            sim = simulate(gamma_true=gamma_true, students=students, seed=seed,
                           concepts=concepts, prereqs=prereqs)
            train, test = split_by_student(sim)
            x_train, delta_train = feature_matrix(train), np.array([r.delta for r in train])
            y_train = np.array([r.correct for r in train], dtype=float)
            prereq_train = np.array([r.prereq_recall for r in train])

            base = fit(x_train, delta_train, y_train, feature_names=FEATURES)
            coupled = fit(x_train, delta_train, y_train, prereq_train, couple=True,
                          feature_names=FEATURES)
            rows.append({
                "gamma_true": gamma_true, "trial": trial,
                "gamma_fitted": coupled.gamma,
                "base": evaluate(test, base, use_coupling=False),
                "coupled": evaluate(test, coupled, use_coupling=True),
                "train_reviews": len(train), "test_reviews": len(test),
            })
    return rows


def schedules(gamma_true: float, students: int = 8, seed0: int = 7700, days: float = 30.0) -> list[dict]:
    """How many reviews a day each scheduler needs to hold 0.8 mean recall.

    Every scheduler gets the same student, the same starting memory and the same
    forgetting process, and none of them can see the student's true half-lives
    except through their own history of right and wrong answers.
    """
    concepts, prereqs, _ = load_concepts()
    results: dict[str, list] = {}
    for student in range(students):
        rng = np.random.default_rng(seed0 + student)
        half_lives = {c: float(rng.lognormal(np.log(2.0), 0.4)) for c in concepts}
        makers = {
            "SM-2": lambda: SM2(),
            "Leitner": lambda: Leitner(),
            "Fixed interval": lambda: Fixed(days=3.0),
            "Half-life, no coupling": lambda hl=half_lives: HalfLifePlanner(
                half_lives=dict(hl), prereqs=prereqs, gamma=0.0, name="Half-life, no coupling"),
            "Half-life, coupled": lambda hl=half_lives: HalfLifePlanner(
                half_lives=dict(hl), prereqs=prereqs, gamma=gamma_true, name="Half-life, coupled"),
            "Half-life, prerequisite-first": lambda hl=half_lives: PrerequisiteFirstPlanner(
                half_lives=dict(hl), prereqs=prereqs, gamma=gamma_true),
        }
        for name, maker in makers.items():
            outcome = reviews_needed(maker, half_lives, prereqs, concepts, days=days,
                                     gamma_true=gamma_true, seed=seed0 + student)
            results.setdefault(name, []).append(outcome)
    summary = []
    for name, runs in results.items():
        budgets = [r["budget"] for r in runs if r["budget"] is not None]
        summary.append({
            "scheduler": name,
            "median_reviews_per_day": float(np.median(budgets)) if budgets else None,
            "mean_reviews_per_day": float(np.mean(budgets)) if budgets else None,
            "students_held": len(budgets),
            "students": students,
        })
    return sorted(summary, key=lambda row: (row["median_reviews_per_day"] is None,
                                            row["median_reviews_per_day"] or 0))


def scarcity(gamma_true: float, budgets=(8, 15, 25), students: int = 6, seed0: int = 7700) -> list[dict]:
    """Mean recall held at budgets too small to keep everything above target.

    This is where knowing about coupling should matter: when you cannot review
    everything, what you choose to drop decides how far the rest falls.
    """
    concepts, prereqs, _ = load_concepts()
    rows = []
    for budget in budgets:
        gathered: dict[str, list] = {}
        for student in range(students):
            rng = np.random.default_rng(seed0 + student)
            half_lives = {c: float(rng.lognormal(np.log(2.0), 0.4)) for c in concepts}
            runners = [
                HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs, gamma=0.0,
                                name="Half-life, no coupling"),
                HalfLifePlanner(half_lives=dict(half_lives), prereqs=prereqs, gamma=gamma_true,
                                name="Half-life, coupled"),
                PrerequisiteFirstPlanner(half_lives=dict(half_lives), prereqs=prereqs,
                                         gamma=gamma_true),
            ]
            for runner in runners:
                outcome = run_schedule(runner, half_lives, prereqs, concepts,
                                       budget_per_day=budget, gamma_true=gamma_true,
                                       seed=seed0 + student)
                gathered.setdefault(outcome["scheduler"], []).append(outcome["mean_recall"])
        for name, values in gathered.items():
            rows.append({"gamma_true": gamma_true, "budget": budget, "scheduler": name,
                         "mean_recall": float(np.mean(values)),
                         "sd": float(np.std(values, ddof=1))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=60)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", default="results.json")
    args = parser.parse_args()
    students = 24 if args.quick else args.students
    trials = 2 if args.quick else args.trials

    concepts, prereqs, _ = load_concepts()
    print(f"{len(concepts)} concepts, "
          f"{sum(len(v) for v in prereqs.values())} prerequisite edges\n")

    rows = recovery(students, trials)
    print("1. Parameter recovery. Does the fit find the coupling it was not told about?\n")
    print(f"   {'true gamma':>11} {'fitted gamma (mean, sd)':>26} {'base MAE':>9} {'coupled MAE':>12} {'base AUC':>9} {'coupled AUC':>12}")
    recovered = []
    for gamma_true in GAMMAS:
        subset = [r for r in rows if r["gamma_true"] == gamma_true]
        fitted = np.array([r["gamma_fitted"] for r in subset])
        base_mae = np.mean([r["base"]["mae"] for r in subset])
        coupled_mae = np.mean([r["coupled"]["mae"] for r in subset])
        base_auc = np.mean([r["base"]["auc"] for r in subset])
        coupled_auc = np.mean([r["coupled"]["auc"] for r in subset])
        print(f"   {gamma_true:>11.2f} {fitted.mean():>13.3f} (sd {fitted.std(ddof=1):>5.3f}) "
              f"{base_mae:>9.4f} {coupled_mae:>12.4f} {base_auc:>9.4f} {coupled_auc:>12.4f}")
        recovered.append({"gamma_true": gamma_true, "gamma_fitted_mean": float(fitted.mean()),
                          "gamma_fitted_sd": float(fitted.std(ddof=1)),
                          "base_mae": float(base_mae), "coupled_mae": float(coupled_mae),
                          "base_auc": float(base_auc), "coupled_auc": float(coupled_auc)})

    null = next(r for r in recovered if r["gamma_true"] == 0.0)
    print(f"\n   The null case. With no coupling in the data, the fit returns "
          f"gamma = {null['gamma_fitted_mean']:.3f} (sd {null['gamma_fitted_sd']:.3f}),")
    print(f"   and the coupled model's held-out MAE is {null['coupled_mae']:.4f} "
          f"against the base model's {null['base_mae']:.4f}.")
    verdict = ("the extension correctly finds nothing"
               if abs(null["gamma_fitted_mean"]) < 0.2 and
               null["coupled_mae"] <= null["base_mae"] + 0.002
               else "the extension is claiming a coupling that is not there, which is a problem")
    print(f"   Reading: {verdict}.")

    print(f"\n2. Reviews a day needed to hold mean recall at or above {HOLD} for 30 days.")
    schedule_results = {}
    for gamma_true in ((0.0, 0.8) if args.quick else GAMMAS):
        table = schedules(gamma_true, students=4 if args.quick else 8)
        schedule_results[gamma_true] = table
        print(f"\n   true gamma {gamma_true:.1f}")
        print(f"   {'scheduler':<26}{'median reviews/day':>20}{'mean':>8}{'students held':>15}")
        for row in table:
            median = f"{row['median_reviews_per_day']:.0f}" if row["median_reviews_per_day"] else "not reached"
            mean = f"{row['mean_reviews_per_day']:.1f}" if row["mean_reviews_per_day"] else "-"
            print(f"   {row['scheduler']:<26}{median:>20}{mean:>8}"
                  f"{row['students_held']:>9}/{row['students']}")

    print("\n3. The same schedulers at budgets too small to hold everything.")
    scarce = []
    for gamma_true in ((0.0, 0.8) if args.quick else GAMMAS):
        rows = scarcity(gamma_true, students=3 if args.quick else 6)
        scarce.extend(rows)
        print(f"\n   true gamma {gamma_true:.1f}: mean recall held over 30 days")
        budgets = sorted({row["budget"] for row in rows})
        print("   " + "scheduler".ljust(32) + "".join(f"{b:>12}/day" for b in budgets))
        for name in dict.fromkeys(row["scheduler"] for row in rows):
            cells = "".join(
                f"{next(r['mean_recall'] for r in rows if r['scheduler'] == name and r['budget'] == b):>16.3f}"
                for b in budgets)
            print(f"   {name:<32}{cells}")

    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump({"recovery": recovered, "raw": rows, "scarcity": scarce,
                   "schedules": {str(k): v for k, v in schedule_results.items()},
                   "settings": {"students": students, "trials": trials, "gammas": GAMMAS}},
                  handle, indent=1)
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
