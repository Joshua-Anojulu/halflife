"""Schedulers to compare against: SM-2, Leitner boxes, and fixed intervals.

These decide *when* to review. The half-life model decides the same thing, so
the comparison is like for like: give each scheduler the same simulated student
and the same review budget, and see which keeps recall highest.

SM-2 is the algorithm Anki's scheduler descends from (Wozniak, SuperMemo 2).
The constants here are the published ones, not tuned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SM2:
    """SuperMemo 2. Quality is binary here: a correct answer is 5, a wrong one 2."""
    name: str = "SM-2"
    ease: dict = field(default_factory=dict)
    interval: dict = field(default_factory=dict)
    repetition: dict = field(default_factory=dict)

    def next_interval(self, concept: str, correct: bool) -> float:
        quality = 5 if correct else 2
        ease = self.ease.get(concept, 2.5)
        repetition = self.repetition.get(concept, 0)
        if quality < 3:
            repetition, interval = 0, 1.0
        else:
            repetition += 1
            interval = 1.0 if repetition == 1 else 6.0 if repetition == 2 else self.interval.get(concept, 6.0) * ease
        ease = max(1.3, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        self.ease[concept], self.interval[concept], self.repetition[concept] = ease, interval, repetition
        return interval

    def due_at(self, concept: str, now: float) -> float:
        return now + self.interval.get(concept, 1.0)


@dataclass
class Leitner:
    """Five boxes with doubling intervals. Correct promotes, wrong resets to box 1."""
    name: str = "Leitner"
    boxes: dict = field(default_factory=dict)
    intervals: tuple = (1.0, 2.0, 4.0, 8.0, 16.0)

    def next_interval(self, concept: str, correct: bool) -> float:
        box = self.boxes.get(concept, 0)
        box = min(box + 1, len(self.intervals) - 1) if correct else 0
        self.boxes[concept] = box
        return self.intervals[box]

    def due_at(self, concept: str, now: float) -> float:
        return now + self.intervals[self.boxes.get(concept, 0)]


@dataclass
class Fixed:
    """Every concept, every N days, regardless of how it went."""
    days: float = 3.0
    name: str = "Fixed interval"

    def next_interval(self, concept: str, correct: bool) -> float:
        return self.days

    def due_at(self, concept: str, now: float) -> float:
        return now + self.days


def make_baselines() -> list:
    return [SM2(), Leitner(), Fixed(days=3.0)]
