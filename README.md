# Halflife

Every spaced repetition system treats items as independent. Course concepts are not. Halflife models a course as a prerequisite graph, lets a concept's memory decay faster when the things it rests on have decayed, and then asks honestly whether that helps.

## The problem

Anki does not know that your recall of the chain rule and your recall of implicit differentiation are the same fact twice. It schedules them separately, drills the dependent concept while the prerequisite has quietly decayed, and you fail the card for a reason the scheduler cannot see.

In the demo: neglect the chain rule for three weeks and implicit differentiation falls from **0.63 to 0.02** predicted recall without being practised any less than before. 39 of 52 concepts in the multivariable calculus graph drop along with it.

## The model

Settles and Meeder (ACL 2016) predict a memory half-life from practice history, `h = 2 ** (theta . x)`, with recall decaying as `p = 2 ** (-delta / h)`. The extension here is one scalar:

```
h_eff(i) = h(i) * product over prerequisites j of p(j) ** gamma
```

`gamma = 0` recovers the base model exactly, so the extension is nested inside the thing it is compared against. That matters: the comparison is honest by construction, and a gamma near zero is a real null result rather than a failure.

## What the study actually asks

Simulating students with coupling built in, then reporting that a coupled model wins, would prove nothing except that the code runs. So the experiment sweeps the **true** coupling strength, including zero, and asks whether the fit finds what was really there.

### 1. The fit recovers a coupling it was never told about

60 simulated students per cell, 5 trials, evaluated on held-out students (the split is on students, never on reviews).

| True gamma | Fitted gamma | Held-out MAE, base to coupled | Held-out AUC, base to coupled |
| --- | --- | --- | --- |
| 0.0 | **0.028** (sd 0.009) | 0.1575 to 0.1558 | 0.9710 to 0.9710 |
| 0.4 | 0.200 (sd 0.012) | 0.1498 to 0.1267 | 0.9750 to 0.9813 |
| 0.8 | 0.542 (sd 0.023) | 0.1463 to 0.1106 | 0.9762 to 0.9863 |
| 1.2 | 0.823 (sd 0.079) | 0.1426 to 0.1094 | 0.9781 to 0.9871 |

The null case is the important row. With no coupling in the data the fit returns 0.028 and the coupled model's held-out error is indistinguishable from the base model's. The extension does not invent a coupling that is not there.

The estimates are **attenuated**: 0.8 reads as 0.54. That is expected and not hidden. The prerequisite recall fed into the fit is itself an estimate, and regressing on a noisy covariate biases its coefficient toward zero. The claim is that the estimate tracks the truth monotonically and vanishes at zero, not that it is unbiased.

### 2. A correction to the published loss, and why it matters here

The original loss has a term penalising the gap between the predicted half-life and the one an observation implies. Settles and Meeder fit on **session recall**, a proportion over several items, so the implied half-life is a real quantity. A single review is **binary**: the implied half-life of a correct answer is unbounded and of a wrong answer is zero, so after clamping that term is almost pure noise.

Left switched on it drags the coupling estimate toward zero. Measured on 40 simulated students, seed 5:

| True gamma | Fitted with the half-life term | Fitted without it |
| --- | --- | --- |
| 0.0 | 0.057 | 0.016 |
| 0.8 | 0.261 | 0.539 |
| 1.2 | 0.247 | 0.790 |

With the term on, a true coupling of 1.2 reads the same as one of 0.8, and the model is blind to the thing it was built to measure. `alpha` therefore defaults to 0 for binary outcomes, and the reason is written in the code where someone will find it.

### 3. Scheduling: a negative result, then a positive one

The half-life planner beats the published baselines clearly. Reviews a day needed to hold mean recall at or above 0.8 for 30 days, median over 8 simulated students:

| Scheduler | gamma 0.0 | gamma 0.8 |
| --- | --- | --- |
| Half-life | **26** | **36** |
| Leitner | 38 | 50 |
| SM-2 | 43 | 46 (held for 6 of 8 students) |
| Fixed interval | 43 | 51 (held for 3 of 8) |

But **the coupling does not help here.** Coupled and uncoupled need the same budget, and the obvious fix, reviewing the prerequisite first because it lifts everything that rests on it, is **worse** than both (40 against 36).

That is worth understanding rather than burying. A scheduler with enough budget to hold everything above target keeps prerequisites fresh as a side effect, so the coupling term sits near 1 and knowing about it changes nothing. The negative result is a consequence of the schedule working.

Starve the budget and it reverses. Mean recall held over 30 days at budgets too small to keep everything up:

| True gamma | Budget | No coupling | Coupled | Prerequisite-first |
| --- | --- | --- | --- | --- |
| 0.0 | 15/day | 0.538 | 0.538 | 0.538 |
| 0.4 | 15/day | 0.196 | **0.362** | 0.112 |
| 0.8 | 15/day | 0.124 | **0.255** | 0.088 |
| 1.2 | 15/day | 0.089 | **0.229** | 0.074 |

Under scarcity the coupled planner roughly doubles the recall held, and at gamma 0 it is identical to the uncoupled one, exactly as it must be. So the finding is conditional and the condition is stated: **coupling helps prediction always, and helps scheduling only when you cannot review everything.** That is the realistic case for a student in a 52-concept course the week before an exam.

Reproduce all of it with `python study.py`, about four minutes.

## The scheduling policy, and why it is not "review what you forgot"

Under this memory model a review multiplies the half-life by `g` when answered correctly and divides by `g` when answered wrongly, so the expected change in log half-life is `(2p - 1) * log(g)`. Reviewing an item whose recall has already fallen below one half is expected to make it **worse**, because the wrong answer is the likely one.

So the planner takes the item closest to slipping from above, not the one most forgotten. If everything has already slipped it takes the one most likely to come back. Finding this mattered: an earlier version chased the most-forgotten item and lost to SM-2.

## Run it

Needs Python 3.11 or newer.

```bash
pip install -r requirements.txt
pytest -q                                      # 19 tests
python study.py                                # the full study, about 4 minutes
python study.py --quick                        # a smaller version, under a minute
python -m http.server 8000                     # then open web/index.html
```

```
data/concepts/calc3.json  52 multivariable calculus concepts, 70 prerequisite edges
model/hlr.py              half-life regression and the coupling term
model/simulate.py         students with a known decay model, so there is ground truth
model/baselines.py        SM-2, Leitner boxes, fixed intervals
schedule/planner.py       the policy, the 30-day run, the budget search
study.py                  recovery, the budget comparison, the scarcity comparison
web/index.html            the graph, coloured by predicted recall
tests/test_model.py       including both the positive and the negative result
```

## Limitations

- **Every number here is from simulation, not from students.** The Duolingo half-life regression dataset, 13 million practice records released with the paper, is the right next step and this repository does not use it. Simulation answers a question real data cannot, which is whether the fit recovers a known coupling. It cannot tell you that human memory actually works this way, and nothing here claims it does.
- **The generative model and the fitted model share a functional form.** Recovery under matched form is the weakest version of the test. Real data would be a much harder one.
- **The prerequisite graph is a teaching judgement.** 52 concepts and 70 edges written by hand from the Stewart chapter 12 to 14 sequence. Reasonable people would draw different edges.
- **The simulated student is simple.** One half-life per concept, a fixed growth factor, no interference between similar concepts, no sleep, no time of day, no motivation.
- **Attenuation is unquantified.** The estimates are biased low and the README says by roughly how much at four points, but no correction is applied.
- **The scheduling comparison gives every scheduler the same budget.** A real student's budget varies, and SM-2 in Anki is tuned differently from the published constants used here.

## Originality note

I have prior work in test preparation: **GrindSAT**, an SAT preparation site. This project shares no code, no data and no subject matter with it. This repository was created after the STEM2Connect hacking period opened on 11 October 2026 and its commit history is the record of that. The overlap is the category "study tool" and nothing else, and I would rather say so here than have anyone wonder.

## AI disclosure

- **Joshua Anojulu**: project lead. Chose the problem and the event, and directs the build.
- **Claude (Anthropic)**: wrote `model/`, `schedule/`, `study.py`, `web/`, the tests, the concept graph and this README. Three things in here came from running the experiment rather than from the original plan, and all three are written up above: dropping the half-life term for binary outcomes, the scheduling policy that does not chase forgotten items, and the finding that coupling helps scheduling only under scarcity. The negative results are reported because they are what the data said.

Earlier project specification and architecture planning: Claude.

## Credits

Burr Settles and Brendan Meeder, "A Trainable Spaced Repetition Model for Language Learning", ACL 2016, for half-life regression. Piotr Wozniak's SuperMemo 2 for the SM-2 baseline constants. Sebastian Leitner for the box system. Concept sequence from the standard multivariable calculus ordering as presented in Stewart.
