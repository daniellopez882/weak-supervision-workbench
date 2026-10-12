# 2. Give MeTaL, FlyingSquid and Snorkel the validation-split class balance; report the unsupervised estimate beside it

Status: accepted

## Context

MeTaL, FlyingSquid and Snorkel's `LabelModel` all need `P(Y)`. Second-order LF overlaps do
not identify it (for two classes any balance can be traded against the LF accuracies), so
there are three options:

* assume a uniform prior;
* estimate it without labels from third-order overlaps, as MeTaL's tensor method does
  (`labelmodels/class_balance.py` implements that from scratch);
* read it off a small labelled split. WRENCH does this: its label-model wrappers take the
  class distribution of the validation split.

During development the tensor estimate recovered the balance of synthetic, conditionally
independent label matrices closely (the tests check it), but on the real WRENCH matrices it
was often far off, most visibly on SMS where it put the minority class in the majority. Real
LFs are correlated (keyword LFs that share words, LFs that only fire together), which breaks
the assumption the tensor method rests on.

## Decision

* In the benchmark, `metal`, `flyingsquid` and `snorkel` receive the validation-split class
  balance, floored so that a class absent from the split keeps a small prior. This is the
  WRENCH protocol and makes the comparison with Snorkel like-for-like.
* `metal_estimated_balance` and `flyingsquid_estimated_balance` run the same models with the
  tensor estimate, and the results file stores both estimates with their L1 error against the
  true training balance. The README reports both rows.
* Dawid-Skene and majority vote take no balance: EM estimates the prior, majority vote has
  none.
* In the web app there is no validation split the user has labelled, so MeTaL and FlyingSquid
  use the tensor estimate when at least three LFs exist and a uniform prior otherwise.

## Consequences

* Headline MeTaL and FlyingSquid numbers use a few hundred gold labels that Dawid-Skene and
  majority vote do not see. The README says so next to the table.
* The estimated-balance rows show what a user without any labels would get, which is the more
  honest number for the app.

## Alternatives rejected

* **Uniform prior everywhere**: simple, but badly wrong on SMS (about one message in seven is
  spam) and unlike the reference protocol, so Snorkel comparisons would not match published
  setups.
* **Estimate only**: would make MeTaL and FlyingSquid look worse for reasons unrelated to the
  models themselves.
* **Tuning the balance on test**: not considered.
