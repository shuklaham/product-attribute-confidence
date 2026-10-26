# How it works and why

Short explanations of each part of the project, in the order of the pipeline. Each section says what the piece does, why it is designed that way, and where it lives in the code. Numbers and decisions are in [plan.md](plan.md); results are in `reports/`.

## 1. The question

A model reads a product listing and fills in an attribute, such as the neckline, together with a stated confidence. A field is filled automatically only when the confidence clears a threshold. That only works if the confidence is honest, including on product categories the threshold was never tuned on.

The study asks whether RL with a calibration reward (M4) keeps confidence more honest under category shift than supervised fine-tuning plus a two-parameter post-hoc fix (M2), and whether the calibration reward itself is what matters (M4 against M3).

## 2. Calibration and the Brier score

**Calibrated** means that among answers stated at 80% confidence, about 80% are right.

The **Brier score** is the mean of (confidence − correct)², where correct is 1 or 0. Lower is better. It rewards two things at once:
- calibration: stated confidence matches the actual hit rate;
- discrimination: right answers get higher confidence than wrong ones.

The M4 reward is correct − (confidence − correct)². It is built on a **proper scoring rule**: honest confidence earns the most reward on average. Suppose an answer is right with probability p and the model states confidence q. The expected reward is

E[R](q) = p · (1 − (1 − q)²) + (1 − p) · (−q²)

Its derivative is 2(p − q), which is zero only at q = p, and the second derivative is −2. So the expected reward peaks when the stated confidence equals the true probability. Overstating or understating both cost reward. `tests/test_output_and_rewards.py` checks this numerically.

M3's reward pays for correctness only. The confidence number earns nothing either way, so nothing pushes it toward honesty. RLCR (arXiv:2507.16806) found that correctness-only RL tends to make models overconfident.

Code: `src/pac/rewards.py`.

## 3. What each comparison isolates

| Comparison | What it shows |
|---|---|
| M0 against M1 | What fine-tuning does to accuracy and to stated confidence |
| M1 against M2 | What a two-number fix does: q′ = sigmoid(a · logit(q) + b), fitted on the calibration split |
| M3 against M4 | The effect of the calibration term alone; everything else is identical |
| M4 against M2 (primary) | Whether RL beats the cheap fix on held-out categories with redacted text |
| M2-logit | A second cheap baseline: the probability the model assigns to its answer letter, rescaled by a temperature |

The Platt map in M2 can correct systematic over- or under-confidence on the data it is fitted to. It cannot reorder confidences, because it is monotone. It also learns the base rates of the calibration split, which is exactly what may break under shift.

## 4. Three ways an evaluation can leak, and the guard for each

1. **The same product in training and test.** Colour variants share photos and descriptions.
   - Families are built with union-find over the `colours` links in each record; the family ID is the smallest product ID in the group.
   - Each family is hashed into one split: sha256 of a salt and the family ID, taken mod 100, picks the bucket. Assignment is therefore deterministic and independent of the order the records are read in.
   - The 12 families that span seen and held-out categories are excluded.
   - Code: `src/pac/splits.py`. Tests: `tests/test_splits.py`.
2. **Held-out categories shaping decisions.** They never reach training, tuning, calibration fitting or threshold choice. They appear only in the final test.
3. **The text stating the answer.** Redaction follows four rules.
   - Rule 1 is enforced by the code's structure: the redaction functions take no label argument, so their output cannot depend on the answer, and a test checks the signature.
   - Rule 3 removes whole clauses ("has a round neck") rather than single words, so the gap does not reveal what was removed.
   - Code: `src/pac/redact.py`. Tests: `tests/test_redact.py`.

## 5. Redaction removes statements, not correlation

After redaction, no attribute word or label phrase survives in any evaluation item; the audit checks every one.

A naive Bayes classifier still beats the category guess on redacted dev text: 81% against 70% for neckline. Naive Bayes scores each label by how often each word appears with it, and picks the label whose words fit best. Its most heavily weighted words show what it relies on:
- brands;
- fabrics, such as "pique" for polo collars;
- styling words.

These are genuine correlates in real product copy, not statements of the answer. So the redacted condition means "no direct statement", and the text-only score on redacted text is the bar the photo has to beat.

Code: `scripts/leak_check.py`.

## 6. Label noise

Catalog labels are not ground truth. When the description names a different label than the catalog field, at least one of them is wrong.

Training on those products would teach the model that correct-looking text is wrong, which is the opposite of using evidence well. So they are excluded from training and from the clean test set, and their rate is reported as a label-noise estimate. For neckline it is 0–26% by category, highest in Kurtis and Tops.

One bug shows why every count needs checking:
- **What went wrong:** brand masking replaced the brand "Lee" inside the word "sleeves". That hid sleeve phrases from redaction and from label matching for every Lee product.
- **How it was caught:** the audit's leftover-word check found 59 hits concentrated in that one brand.
- **The fix:** brands now match whole words only, with a regression test.

Code: `src/pac/text.py`.

## 7. The hand check

The exclusion rule in section 6 relies on a phrase matcher. A sample of 100 flagged contradictions answers two separate questions.

1. **Does the description really name a different label?** This is a reading task, and it measures the matcher's precision. 95 of 100 were genuine. The 5 misreads led to four pattern fixes, each with a regression test.
2. **Which source is right, the catalog or the description?** This needs the photo. Both turned out to be wrong regularly: the description in 37 cases and the catalog label in 29. So excluding contradictions is safer than trusting either source.

Three design choices keep the check honest:
- **Test splits are never sampled.** If a review changed the exclusion rule, test items we had looked at could re-enter the test set.
- **The sample is frozen** in `annotations/hand_check_queue_v0.csv`, so later matcher fixes do not reshuffle it.
- **Every verdict records its reviewer.** Claude's verdicts come from a vision-language model, the same kind of model the study measures, so they have unknown error. The author's confirmations give an agreement rate that says how far to trust them.

The result: in a spot check of 20 of Claude's decisive verdicts, the author agreed on 18. On 20 items, the 95% Wilson interval for that 90% runs from 70% to 97%. The Wilson interval is a confidence interval for a proportion that behaves well with small counts. Both disagreements were round versus boat necklines, so the boundary between those labels is fuzzy even for careful reviewers.

The check also exposed a definitional gap in sleeve length. Roll-up sleeves photographed rolled up look short, but the garment is long. Cap sleeves sit between short and sleeveless.

A label that depends on a convention rather than on the evidence makes every method look miscalibrated for reasons that have nothing to do with the method. So products whose text describes roll-up sleeves are excluded from the sleeve-length task, through an `exclude_if` rule in the ontology. A text rule cannot see roll-ups that show only in the photo, and those remain as label noise.

Code: `src/pac/review.py`, `scripts/review_queue.py`, `app/review.py`.

## 8. Category shift has two parts

- **New kinds of products.** A kurta does not look like a T-shirt.
- **Changed label frequencies.** Mandarin collar is 1% of training necklines but 13% of held-out ones. Three-quarter sleeves go from 1.5% to 20%.

Calibration cares about the second part because a confidence map fitted on training-category data carries the training base rates. When the frequency of an answer changes, the same stated 0.9 can correspond to a different actual hit rate. Per-label reliability results separate the two effects.

## 9. Reproducibility guards

- **Fingerprinted manifests.** `manifests/v0` stores product IDs, splits and label decisions, with a fingerprint of the source files and the hash of the label set. `scripts/build_examples.py` refuses to run if either has changed, so results never silently mix data versions.
- **Strict output parsing.** Invalid output counts as wrong and is never repaired, so a model cannot look better by producing answers that a lenient parser quietly fixes. Code: `src/pac/output.py`.
