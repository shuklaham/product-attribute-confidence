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

1. **The same product in training and test.** Colour variants share photos and descriptions, and the catalog also lists some garments twice under different IDs with no colour link.
   - Families are built with union-find over two kinds of link: the `colours` links in each record, and byte-identical photo files (sha256 of the original JPEG). The family ID is the smallest product ID in the group.
   - The photo links were added after an independent review: with colour links alone, 246 groups of identical photos crossed splits, and 6–7% of seen-test items had the exact photo and label in training. That would have inflated seen-test results and the shift estimate.
   - Near-duplicates, such as a re-shot or re-encoded photo, are not caught; a tight perceptual-hash rule produced false matches on plain T-shirts.
   - Each family is hashed into one split: sha256 of a salt and the family ID, taken mod 100, picks the bucket. Assignment is therefore deterministic and independent of the order the records are read in.
   - The 12 families that span seen and held-out categories are excluded.
   - Code: `src/pac/splits.py`. Tests: `tests/test_splits.py`.
2. **Held-out categories shaping decisions.** They never reach training, tuning, calibration fitting or threshold choice. They appear only in the final test.
3. **The text stating the answer.** Redaction follows four rules.
   - Rule 1 is enforced by the code's structure: the redaction functions take no label argument, so their output cannot depend on the answer, and a test checks the signature.
   - Rule 3 removes every clause that names a label, contains a cue or mentions the attribute, whole ("has adjustable straps") rather than word by word, so neither a gap nor a remnant reveals what was removed. An earlier version dropped a clause only when an attribute word was left after removing phrases, so "has adjustable straps" became "has adjustable", and "adjustable" alone predicted sleeveless.
   - In the title and the opening noun phrase, dropping the clause would also drop the product type. There a mention is removed together with the words that describe it, back to the earliest describing word within three words: "Three Fourth Sleeve Kurta" becomes "Kurta", where the earlier rule left "Three Kurta".
   - Code: `src/pac/redact.py`. Tests: `tests/test_redact.py`.

## 5. Redaction removes statements, not correlation

After redaction, no attribute word, label phrase or cue survives outside a brand name in any evaluation item; the audit checks every one. The word list includes what product copy really contains:
- abbreviations in brand titles: SS and HS for short sleeves, LS and FS for long, S/L and Sless for sleeveless;
- misspellings such as "sleevless", "leeveless", "sphagetti" and "necklne";
- redaction-only cues, words that point to a value without naming one ("armholes" is in 78% of cases a sleeveless item), too loose for label matching but removed from the redacted text.

A checker that only tests the word list cannot find what the list is missing. The review found these gaps with a separate scan using a broader list, and that scan now runs too.

A naive Bayes classifier still beats the category guess on redacted dev text: 82% against 71% for neckline. Naive Bayes scores each label by how often each word appears with it, and picks the label whose words fit best. The words that move it most show what it relies on:
- product types, such as "tee" for round necks;
- styling words, such as "placket" for polo collars and "cuffs" for long sleeves;
- the model's measurements ("5'8", "33"), which mark women's tops.

These are genuine correlates in real product copy, not statements of the answer. So the redacted condition means "no direct statement", and the text-only score on redacted text is the bar the photo has to beat.

Code: `scripts/leak_check.py`.

## 6. Label noise

Catalog labels are not ground truth. When the description names a different label than the catalog field, at least one of them is wrong.

Training on those products would teach the model that correct-looking text is wrong, which is the opposite of using evidence well. So they are excluded from training and from the clean test set, and their rate is reported as a label-noise estimate. For neckline it is 2–26% by category, highest in Kurtis and Tops. They are still built for the test splits, marked `clean = false`, so results can also be reported on all labeled items.

Only the text the model sees counts: the title and description, never the variant name.

One bug shows why every count needs checking:
- **What went wrong:** brand masking replaced the brand "Lee" inside the word "sleeves". That hid sleeve phrases from redaction and from label matching for every Lee product.
- **How it was caught:** the audit's leftover-word check found 59 hits concentrated in that one brand.
- **The fix:** brands now match whole words only, with a regression test.

Code: `src/pac/text.py`.

## 7. The hand check

The exclusion rule in section 6 relies on a phrase matcher. A sample of 100 flagged contradictions answers two separate questions.

1. **Does the description really name a different label?** This is a reading task, and it measures the matcher's precision. 95 of 100 were genuine. The 5 misreads led to four pattern fixes, each with a regression test.
2. **Which source is right, the catalog or the description?** This needs the photo. Both turned out to be wrong regularly: the description in 37 cases and the catalog label in 29. So excluding contradictions is safer than trusting either source.

Design choices that keep the check honest:
- **Test splits are never sampled.** If a review changed the exclusion rule, test items we had looked at could re-enter the test set. The app no longer shows test splits at all, and any item reviewed earlier that now falls in a test split (8 in v1) is excluded from every test set.
- **The sample is frozen** in `annotations/hand_check_queue_v0.csv`, so later matcher fixes do not reshuffle it.
- **Every verdict records its reviewer.** Claude's verdicts come from a vision-language model, the same kind of model the study measures, so they have unknown error. The author's verdicts give an agreement rate that says how far to trust them.
- **The author's verdicts are blind.** The app hides Claude's verdict until the author has saved their own, nothing is pre-selected, and each verdict records whether it was blind.

The last point was added after the review found a flaw. In the first spot check the app showed Claude's verdict and pre-selected it. The author agreed on 18 of 20 (95% Wilson interval 70–97%), but agreement measured with the answer on screen is anchored, so 18 of 20 is a confirmation rate and an upper bound on independent agreement. The Wilson interval is a confidence interval for a proportion that behaves well with small counts. Both disagreements were round versus boat necklines, so that boundary is fuzzy even for careful reviewers. Blind agreement is measured on the items the author has not yet reviewed.

The check also exposed a definitional gap in sleeve length. Roll-up sleeves photographed rolled up look short, but the garment is long. Cap sleeves sit between short and sleeveless.

A label that depends on a convention rather than on the evidence makes every method look miscalibrated for reasons that have nothing to do with the method. So products whose text describes roll-up sleeves are excluded from the sleeve-length task, through an `exclude_if` rule in the ontology. The roll-up phrase must sit near "sleeve" or "cuff", so "denims rolled up to the ankle" does not count, and the catalog's Sleeve Styling field also marks roll-ups. Roll-ups that show only in the photo remain as label noise.

Code: `src/pac/review.py`, `scripts/review_queue.py`, `app/review.py`.

## 8. Category shift has two parts

- **New kinds of products.** A kurta does not look like a T-shirt.
- **Changed label frequencies.** Mandarin collar is 1.0% of training necklines but 12.7% of held-out ones. Three-quarter sleeves go from 1.3% to 19.7%. These shares are before the training cap; the cap raises them to 1.4% and 2.0% of what the model trains on.

Calibration cares about the second part because a confidence map fitted on training-category data carries the training base rates. When the frequency of an answer changes, the same stated 0.9 can correspond to a different actual hit rate. Per-label reliability results separate the two effects.

## 9. Reproducibility guards

- **Fingerprinted manifests.** `manifests/v1` stores product IDs, splits, label decisions and photo hashes, with a fingerprint of the source files, the hash of the label set, the hash of the review annotations and the hash of each manifest file. `scripts/build_examples.py` refuses to run if any of them changed, and writes a build record with the hash of every example file; each model run records the hash of the example file it read. The code version is the git commit in each run record.
- **Clean text.** Runs of garbled characters left in some listings are removed before anything else; one description was 27,000 characters of them, longer than the model's context. Code: `src/pac/text.py`.
- **Strict output parsing.** Invalid output counts as wrong and is never repaired, so a model cannot look better by producing answers that a lenient parser quietly fixes. Code: `src/pac/output.py`.
- **One prompt path.** Evaluation builds prompts with the model's own chat template through its Hugging Face processor, the path training uses, so both see identical prompt strings. Each run records a hash of every prompt, the model revision, the package versions and the git commit. Code: `scripts/predict.py`.

## 10. Measuring confidence

Each metric answers a different question, and a method can win on one while losing on another.

| Metric | Question | Detail |
|---|---|---|
| Accuracy | How often is the answer right? | Pooled, and averaged over category–attribute groups so T-shirts do not dominate |
| Brier score | How far is the confidence from the outcome? | Mixes calibration and discrimination; the score M4 is trained on |
| ECE | Does stated confidence match the hit rate? | 10 equal-width bins; the count-weighted gap between confidence and accuracy |
| AUROC | Are right answers more confident than wrong ones? | Ranking only: 0.5 is chance, 1 is perfect |
| Risk–coverage area | If the most confident items are filled first, how many errors pile up? | The mean error rate over every fill level; lower is better |
| Coverage at the threshold | What share is filled at the 5% error target? | The threshold is picked on the calibration split, then frozen |

**Why calibration alone is not enough.** A model that states its average accuracy for every item has an ECE of zero, but it cannot tell easy items from hard ones (AUROC 0.5), so it is useless for auto-fill.

**What the Platt map can and cannot change.** With a > 0 the map is increasing, so it keeps the order of confidences. M2 then has M1's AUROC, risk–coverage curve and auto-fill coverage: the same items clear the threshold, and only the numbers attached to them change. (Stated values above 0.995 tie after clipping, see below, so that holds unless the model uses several of them.) M2 can improve only Brier and ECE. If M4 fills more than M2 at the same error, RL must have changed which items the model is confident about.

**Two edge cases.**
- Invalid outputs count as wrong and are never auto-filled, so they rank last in the risk–coverage area. That makes an invalid output score like a wrong answer given the lowest confidence: no better than an honest low-confidence answer, but better than a confidently wrong one. The same holds for AUROC and ECE, which skip invalid outputs, and for the Brier score on valid outputs only. The Brier score with each invalid output counted as a loss of 1, the worst possible, is the one variant an invalid output can never improve, so the primary comparison uses it, and the invalid rate is reported beside every ranking metric. An earlier draft of this section claimed an invalid output could never improve any score; the review showed that was wrong.
- Stated confidences pile up on a few values such as 0.9 and 0.95; CalibSFT reports the same. A threshold cannot split a tie, so the curve has one point per distinct value, and the area is averaged over a random order within each tie.

**The threshold.**
- Rule: the lowest threshold at which the accepted items on the threshold half of the calibration split (409 neckline and 564 sleeve items) have at most 5% errors. The lowest qualifying threshold fills the most.
- Picking the best of many thresholds on a few hundred items tends to land on one whose true error is a little above 5%. The seen test shows that overshoot without any shift, so the difference between the held-out and seen results measures the shift itself.

**Fitting the cheap fixes.** Both are fitted on the map-fitting half of the calibration split, then frozen.
- Platt map: q′ = sigmoid(a · logit(q) + b), with a and b chosen to minimize cross-entropy by Newton's method. a stays at or above 0, so the map never reverses the order. If the best fit would need a < 0, meaning higher stated confidence goes with lower accuracy, the map becomes the constant base rate. If every stated value is the same, the data say nothing about the slope, so a stays at 1 and only the level moves.
- 0 and 1 have no finite logit, so stated confidence is clipped to [0.005, 0.995] first. Models state confidence in steps of 0.01 or coarser, so a stated 1.0 is read as half a step above 0.99. The clip is fixed in advance because it matters: with many stated 1.0s it decides where that cluster sits on the logit axis, and the fit follows.
- Temperature: the answer-letter probabilities become softmax(log-probabilities / t), with t chosen to minimize the negative log-likelihood of the correct letters by golden-section search. A t above 1 softens overconfident probabilities. The chosen letter never changes, so accuracy is unchanged.

**Intervals.** Differences between methods come from a paired bootstrap over product families. Each resample draws families with replacement, keeps every row of a drawn family together, and scores both methods on the same draw. Colour variants are near-duplicates, so resampling single items would overstate the evidence and make the intervals too narrow. M2's map is itself an estimate from a few hundred calibration items, so for M2 each draw also refits the map on a resample of the calibration families. A statistic that is undefined on some draw, such as AUROC on a draw with no wrong answers, stops the bootstrap with an error instead of quietly shrinking the interval.

Code: `src/pac/metrics.py`, `src/pac/calibration.py`. Tests: `tests/test_metrics.py`, `tests/test_calibration.py`.

## 11. Reading the answer-letter probability

M2-logit needs the probability the model gives each option letter, not just the letter it writes.

- The model writes `{"choice": "B", ...}`. The Qwen tokenizer makes the option letter a token of its own right after `"choice": "`. That holds in every layout checked: spaced, compact, fenced in a code block, and spread over several lines.
- At that position, vLLM returns the log-probability of all 12 letter tokens (`logprob_token_ids`), not just the top few, so no offered letter is ever missing.
- The values are read before any sampling adjustment (vLLM's raw mode), so decoding settings cannot change them.
- The probabilities are renormalized over the letters actually offered: 9 for neckline, 4 for sleeve length. The chosen letter's share is the confidence, before and after temperature scaling.
- The same strict parse decides validity for both sources of confidence, so M1, M2 and M2-logit are scored on the same answers. The exception is a valid output whose choice position cannot be read, for example when a tab before the quote merges it into the letter token; such items get no letter confidence, rank last in the letter-based area with their real correctness, and are counted in the tables.
- The parser also rejects a repeated key: JSON would keep the last value, while the letter probabilities are read where the first one was written.

Code: `src/pac/inference.py`, `src/pac/evaluation.py`, `scripts/predict.py`. Tests: `tests/test_inference.py`.
