# 1. Treat abstain as an outcome in Dawid-Skene, and start silent rows from the covered class mix

Status: accepted

## Context

Dawid-Skene was written for crowd workers who give every item a label or skip it at random.
Labelling functions behave differently in two ways that matter for EM:

1. **Most LFs are unipolar.** Every LF shipped with YouTube, SMS, TREC and AG News votes for
   one class or abstains (Yelp is the exception: most of its LFs vote both ways). If abstains
   are treated as missing at random, a unipolar LF's
   confusion matrix conditioned on voting is the same for every true class (it always says its
   one class), so it carries no information and the posterior falls back to the prior. On
   YouTube the missing-at-random variant converged in a handful of iterations to roughly the
   class prior during development.
2. **Silence is common and not random.** On SMS most rows get no vote at all, and the spam LFs
   fire mostly on spam. With abstain modelled as an outcome, EM is free to explain "nobody
   fired" as a class of its own. Starting those rows from a uniform posterior (the usual soft
   majority-vote initialisation) pushed EM there: it labelled most silent rows as spam, far
   below the majority-class baseline on test.

## Decision

* `DawidSkene` models `P(lambda_j = o | Y = y)` over `o in {0..k-1, abstain}` by default
  (`abstain_as_outcome=True`), so *when* a unipolar LF fires is evidence.
* EM is initialised with soft majority vote on rows that have votes, and rows without votes
  start at the class mix of the covered rows, weighted by how often each vote pattern occurs.
  Rows labelled by a person are clamped to one-hot for the whole run.

`tests/test_labelmodels.py::test_dawid_skene_starts_uncovered_rows_from_covered_class_mix`
pins the second point on a synthetic SMS-like matrix.

## Consequences

* The model is still unsupervised and still the only label model here that can absorb human
  labels directly, which is why the active-learning loop uses it (see ADR 3).
* Coverage becomes part of each LF's parameters, so an LF whose firing rate differs between
  classes is rewarded for it. That is a real signal on WRENCH, but it also means a label model
  fitted on one data distribution transfers less well to another with different coverage.
* `abstain_as_outcome=False` remains available for multi-polar LFs and crowd-style data.

## Alternatives rejected

* **Missing-at-random abstains** (classic Dawid-Skene): uninformative for unipolar LFs, as
  above.
* **Uniform start for silent rows**: the failure mode on SMS described above.
* **Fitting EM on covered rows only**: avoids the degenerate class but leaves the abstain
  probabilities unidentified for silent rows, which then get only the prior at prediction time,
  and makes the posterior of a row depend on whether it was in the fitting set.
