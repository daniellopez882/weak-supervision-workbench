# 3. Shape of the active-learning experiment

Status: accepted

## Context

The spec asks for learning curves at 25, 50, 100, 250 and 500 human labels for three
selection strategies against random, with five seeds per dataset, and the whole benchmark
should fit in roughly half an hour on a 6-core desktop CPU. The expensive part is the end
model: TF-IDF plus logistic regression on AG News' 96000 training rows takes several seconds
per fit, and a naive loop would refit it after every selected batch.

## Decision

* **Label model in the loop: Dawid-Skene with human labels clamped.** It is the only label
  model here whose fit can take gold labels, so every batch changes the posteriors that the
  margin strategy reads and the soft labels the end model trains on. Other label models can be
  selected (`LoopConfig.label_model`); for them human labels only replace the targets.
* **Batches of 25, end model only at checkpoints.** The label model is refitted after every
  batch (cheap); the end model is retrained at 0, 25, 50, 100, 250 and 500 labels, warm-started
  from the previous checkpoint.
* **Pool capped at 10000 rows.** Yelp and AG News draw a fresh pool per seed; all strategies
  under one seed share that pool and the same starting label model, so differences between
  strategies are paired. The smaller sets use their whole training split.
* **Summary number: normalised AULC over 25..500 on a linear label axis.** The trapezoidal area
  divided by the budget range is the curve's average height, so it reads as an accuracy. The
  0-label point is identical for all weak-supervision strategies and is shown in the curve
  tables, not in the area. Each strategy is also reported as the mean paired difference to
  random and the number of seeds on which it beat random.
* **A human-labels-only reference curve** (random selection, end model trained on the human
  labels alone) shows what the weak labels are worth at each budget.
* **The simulated oracle answers from the gold training labels** and counts queries; all
  randomness comes from generators derived from the seed, and a test asserts that two runs
  with the same seed select the same rows and report the same numbers.

## Consequences

* The curves describe this end model and this label model; a stronger end model (fine-tuned
  transformer) could change the ranking of strategies.
* The pool cap means Yelp and AG News curves are about a 10000-row pool, not the full
  training split; the label-model and end-model tables still use the full splits.
* With five seeds, small nAULC differences are within noise; the README reports standard
  deviations and win counts rather than claiming a winner.

## Alternatives rejected

* **Retrain the end model after every batch**: about four times more end-model fits for points
  nobody reads.
* **Log-scaled label axis for AULC**: weights the 25..100 region more, where strategies
  differ most, but makes the number harder to read as an accuracy. Per-budget tables keep
  that detail visible instead.
* **Full pools for Yelp and AG News**: the core-set strategy and the end model dominate the
  run time at 96000 rows and would push the benchmark well past its time budget.
