# Run log

One entry per working session: what ran, what it cost, what it showed, and what changed.

## 2026-10-25: Milestone 1, data foundation

**Ran**
- Downloaded the official Kaggle dataset (24.8 GB) and extracted its 44,446 product records.
- Built ontology v0, the `pac` library, the frozen v0 manifests, model-facing examples in both text conditions, the v2 audit, and the CPU leak check.
- Checked and resized photos: 15,709 of 15,712 in-scope products have a photo; 3 are missing from the zip, none are unreadable.
- 64 tests pass.

**Cost:** no GPU used.

**Findings**
- The official records are byte-identical to the GitHub mirror used for the first audit (source fingerprint `f1352f4f7be4`), so earlier numbers stand.
- Bug fixed: brand masking matched inside words (the brand "Lee" inside "sleeves"), which hid sleeve phrases from redaction and from label matching. Brands now match whole words only, with a regression test.
- Tightening the phrase matcher changed the label-noise estimate. Most earlier "contradictions" in Dresses and Nightdress were strap phrases and the "U.S. Polo" brand. The neckline rate is still 25% for Tops and 26% for Kurtis, against 2% for T-shirts and 4% for Kurtas.
- Included labels: 20,068.
  - Neckline: 4,704 train, 714 dev, 664 seen test, 1,234 held-out test.
  - Sleeve length: 7,304 train, 1,111 dev, 1,058 seen test, 1,449 held-out test.
  - Training caps of 1,500 per (category, label) cell then apply when examples are built.
- No attribute word or label phrase survives redaction in any evaluation item.
- Leak check on dev: redacted text still beats the category guess (neckline 81% against 70%; sleeve balanced accuracy 62% against 46%). The words it relies on are brands, fabrics and styling, not statements of the attribute, after adding "crew", strap words, "armhole" and "mid-arm". Word list v0 is frozen.

**Changed**
- Plan: refined the redaction check and recorded the photo rule (decision log, 2026-10-25).

**Next**
- Hand check of 100 sampled contradictions in the review app.
- Milestone 2: Runpod pod and volume, vLLM inference, metrics, M0 on dev.

## 2026-10-25: Hand check of description–label contradictions

**Ran**
- Drew a frozen sample of 100 contradictions from training-side splits (`annotations/hand_check_queue_v0.csv`). The test splits stay unseen.
- `scripts/review_queue.py` built a text sheet and 17 photo contact sheets. Claude read every description and looked at every photo.

**Cost:** no GPU used.

**Findings**
- Text half, the phrase matcher's precision: 95 of 100 contradictions are genuine. The 5 misreads were "spread collar" on a polo jersey, a two-piece set with a sleeveless inner layer, "armholes" on a sleeved top, "deep round neck" (a scoop), and "three quarter pants". Four patterns were fixed, with regression tests.
- Photo half, Claude's verdicts on the 95 genuine contradictions:
  - The description is wrong in 37.
  - The catalog label is wrong in 29.
  - 29 can't be told from the photo, 14 of them roll-up sleeves.
- Both sources err, so excluding contradictions is safer than trusting either one.
- The author had reviewed 16 contradictions in the app before the queue was rebuilt. Two overlap the new sample: one agrees, and one is a roll-up sleeve that the author labeled by construction (long) and Claude marked "Can't tell".
- After the pattern fixes, 26 more labels are included (20,094 in total). Redaction still leaves no label phrase in any evaluation item. None of the 7 test items the author inspected re-enters a test set; all remain excluded as genuine contradictions.

**Decision:** products whose text describes roll-up sleeves are excluded from the sleeve-length task in every split.
- The rule removes 590 sleeve labels: 559 in seen categories and 31 held-out.
- Included labels: 19,558.
- Redaction still leaves no label phrase in any evaluation item.
- The text rule catches 8 of the 14 roll-ups in the hand check. The rest show only in the photo and remain as label noise.

**Spot check, by the author.** The sample was 20 of Claude's decisive verdicts: the first 10 "Description is wrong" and the first 10 "Catalog label is wrong" calls, in queue order.
- Agreement: 18 of 20, or 90%, with a 95% Wilson interval of 70% to 97%.
- "Description is wrong": 10 of 10. "Catalog label is wrong": 8 of 10.
- Both disagreements are round versus boat necklines, where Claude called a wide neckline boat and the author called it round.
- Across all 26 queued items the author reviewed, they agree on 22.
- Claude's verdicts are reported as an AI review checked against the author at this rate.

**Next**
- Milestone 2.

## 2026-10-25: Milestone 2, local part

**Ran**
- Wrote the metrics, calibration, inference and evaluation modules, with tests; 101 pass.
- Wrote `scripts/predict.py` (vLLM, on the GPU pod) and `scripts/evaluate.py` (metric tables). A smoke test with stand-in modules for vLLM, transformers and torch ran both end to end on 20 dev items.
- Checked the Qwen3-VL tokenizer and chat template at the pinned model revision (`ebb281ec`).

**Cost:** no GPU used.

**Findings**
- In every output layout checked, the option letter is a single token right after `"choice": "`, so letter probabilities can be read at one position.
- vLLM 0.32 can return log-probabilities for a fixed list of token IDs at each position. That gives every offered letter, with no top-k cutoff.
- The model's chat template adds no system prompt. Its generation config defaults to sampling (temperature 0.7, top-p 0.8). vLLM applies those defaults only when no sampling parameters are passed, and the scripts always pass greedy settings.
- vLLM 0.32.0 pins torch 2.13.0, whose wheels need CUDA 13, so the pod's host needs NVIDIA driver 580 or newer.
- The plan's label table predated the hand-check fixes and the roll-up exclusion. It now matches `manifests/v0`; for example, three-quarter sleeves are 1.3% of training sleeves, not 1.5%.

**Changed**
- Plan: decision log rows for how metrics treat invalid outputs and ties, the threshold rule, and reading letter probabilities with pinned versions; corrected label table.
- Dependencies: `numpy>=2.0` instead of `>=2.5.4`, because vLLM's numba needs numpy below 2.5. A `gpu` dependency group pins vLLM 0.32.0 for Linux.

**Next**
- GPU part of Milestone 2: pod and volume, upload examples and photos, a 16-item smoke run, M0 on dev in both conditions, and the text-only redaction check.

## 2026-10-25: Independent review and data v1

**Ran**
- Two independent reviews of all work so far, one of the data pipeline and leakage, one of the evaluation and inference code. Every finding was reproduced before it was fixed.
- Rebuilt the data as v1: photo hashes in `manifests/v1/photos.csv`, the manifests, the examples (`data/processed/v1`, with a build record), the audit and the CPU leak check.
- Ran the review app headlessly in both modes. 130 tests pass.

**Cost:** no GPU used. The Runpod account now has the laptop's SSH key registered; creating the network volume was refused because the balance is under $5.

**Findings, data side**
- **Duplicate listings crossed splits.** Some garments are listed twice under different IDs with no colour link. In v0, 246 groups of byte-identical photos spanned two splits, and 6–7% of seen-test items had the same photo and label in training. The held-out test was not affected.
- **Statements survived redaction.**
  - Rule 3 dropped a clause only if an attribute word was left, so "has adjustable straps" became "has adjustable".
  - The hand-check fix had removed "armholes" from redaction along with matching.
  - Misspellings, brand abbreviations (SS, HS, LS, FS, S/L, Sless) and title fragments ("Three Kurta") also survived.
  - Measured with the v1 word list, 5.3% of held-out redacted sleeve items still contained a statement or cue, against 1.3% in the seen test. The audit could not see it, because it only checked the word list itself.
- **The spot check was anchored.** The app showed Claude's verdict and pre-selected it, so 18 of 20 is a confirmation rate, not independent agreement.
- **Garbled text.** Runs of mis-decoded characters in 37 listings made 16 prompts longer than the model context.
- **Smaller issues:**
  - The text check also read the variant name, which the model never sees; with the v1 word list it would change 223 checks.
  - Families that mixed a kept and a dropped value were not treated as disagreeing.
  - 30 roll-up exclusions had no "sleeve" or "cuff" near any roll-up phrase, and the rule missed 9 included items that the catalog's own field marks as roll-ups.
  - 8 items reviewed earlier sit in test splits under v1.
  - The plan's "all labeled items" test grid could not be built.

**Findings, evaluation side**
- **Merged runs.** The table script merged different models or splits into one row, and accepted unfinished runs.
- **A wrong claim about invalid outputs.** The docs claimed an invalid output can never improve a score. In fact, replacing a confidently wrong answer with an invalid one improves the risk–coverage area, AUROC, ECE and the valid-only Brier score.
- **Mismatched item sets.** Evidence sensitivity measured its two drops on different items.
- **Smaller issues:**
  - Duplicate JSON keys were accepted.
  - The Platt fit depended on an unregistered clip and failed silently on degenerate input; the temperature fit failed silently on bad input.
  - A bootstrap interval could silently turn NaN.
  - A right answer without a letter confidence counted as an error in the letter-based area.
- **Checked and sound:**
  - The Platt and temperature fits match SciPy to 1e-15.
  - AUROC, ECE and risk–coverage tie handling are correct.
  - The vLLM 0.32.0 and transformers 5.19.0 calls are right, and so are the tokenizer claims.
  - v0 manifests and examples rebuild byte for byte.
  - Families, held-out isolation, the prompt's fields and the option order are sound.

**Changed**
- **Data v1:**
  - Families also join byte-identical photos. 513 photo groups cover 1,120 products; 291 products changed split; families went from 12,452 to 11,881. No photo is in two splits.
  - Redaction v1 drops whole clauses and covers the abbreviations, misspellings and cues above; titles lose the words that describe a mention.
  - The text check reads only the title and description.
  - Dropped values count in family disagreement.
  - The roll-up rule needs "sleeve" or "cuff" nearby and also reads the catalog field.
  - Garbled runs are removed, so the longest prompt is now 608 text tokens.
  - Reviewed items in test splits are excluded from every test set.
  - The test splits now include the items excluded only as label noise, marked `clean = false`.
- **Redaction checks, v1:**
  - No attribute word, phrase or cue survives outside a brand name in any evaluation item; 56 neckline items keep "Polo" inside a brand name.
  - CPU leak check on dev: redacted neckline text scores 82% against a category guess of 71%. Redacted sleeve text scores 84% against 86%, but 61% against 46% in balanced accuracy.
  - The words it relies on are product types, styling ("placket", "cuffs") and model measurements. Word list v1 is frozen.
- **Included labels: 19,535.** Built items, clean:

  | Split | Neckline | Sleeve length | Notes |
  |---|---:|---:|---|
  | Train | 3,280 | 4,437 | after the 1,500 cap; 4,693 and 6,944 before it |
  | Dev | 710 | 1,063 | |
  | Calibration, fitting half | 327 | 482 | |
  | Calibration, threshold half | 409 | 564 | |
  | Seen test | 667 | 1,020 | plus 101 noisy items |
  | Held-out test | 1,232 | 1,424 | plus 197 noisy items |

  Cowl neck now has 29 training labels, under the 30 needed, so like Square neck and Shirt collar it is offered but never scored.
- **Review app:**
  - Claude's verdict stays hidden until the author saves theirs, and nothing is pre-selected.
  - Verdicts record whether they were blind.
  - Test splits never appear.
- **Evaluation code:**
  - Each prediction file is one run, and rows are keyed by run, split and subset.
  - Evidence sensitivity joins the two conditions and uses one item set.
  - The parser rejects repeated keys.
  - The fits refuse bad input.
  - `predict.py` checks every prompt's length before using the GPU, and its run record hashes the photos too.
- **Protocol, fixed before any training:**
  - The primary comparison uses the Brier score with invalid outputs counted as loss 1, pooled over both attributes.
  - M2's map is refitted in every bootstrap draw.
  - Stated confidence is clipped to [0.005, 0.995] before the Platt fit.

**Corrections to earlier entries**
- **Roll-up rule:** "removes 590 sleeve labels" counted labels already excluded for other reasons. 536 of the 590 were otherwise included.
- **Hand check:** of the 43 sampled sleeve contradictions, 40 were genuine. Roll-up sleeves came up in 15 of them, 14 of which Claude could not decide from the photo. Cap versus sleeveless cases were 7, not 4.
- **Leak check:** re-running it on v0 gives a sleeve balanced accuracy of 63%, not 62%.
- **Redaction residual:** "no label phrase survives" was true only outside brand names.
- **Photos:** not every photo is 1800×2400; 4,562 are 1080×1440.

**Next**
- Add Runpod credit (author), then the GPU part of Milestone 2 on the v1 examples.
- A blind review of the remaining hand-check items in the app (author), to measure agreement with Claude independently.
- Before Milestone 4: decide the RL data mix, since the training cap changes the label mix that RL calibrates against.

## 2026-10-26: Milestone 2, base model on dev

**Ran**
- One A40 pod (Runpod Secure Cloud, EU-SE-1, CUDA 13.2 host) with vLLM 0.32.0, torch 2.13.0 and transformers 5.19.0 from `uv.lock`, at commit `2931c4d`.
- A 16-item smoke test, then three runs on the v1 dev split:
  - the base model with photo and text, in both text conditions (3,546 examples);
  - text only, on redacted text (1,773);
  - photo only (1,773).
- The pod had no network volume, because EU-SE-1 offers none and CA-MTL-1 had little A40 stock. Results were copied back, and the pod was deleted.
- Tables: `reports/m0_dev.md`.

**Cost:** about 16 minutes of pod time at $0.59/hr, about $0.16. Runpod's billing records had not posted when checked.

**Findings**
- **Accuracy is above the plan's 90% saturation line on redacted dev.** Neckline is 95.5% (95% interval 93.6–97.2%) and sleeve length 93.6% (91.9–95.2%). With full text they are 97.2% and 95.9%.
- **The photo carries the answer.** Photo only scores 94.8% and 93.4%. Text only on redacted text scores 76.2% for neckline, against a category guess of 70.8%, and 73.5% for sleeve length, below the category guess of 85.5%.
- **No invalid outputs** in any run.
- **The stated confidence carries no information; the letter probabilities do.**
  - The model states 0.95 on 99–100% of its answers with the photo, so the stated confidence cannot rank them: AUROC 0.50–0.52.
  - ECE is low (0.002–0.022) only because accuracy happens to sit near 0.95.
  - The probability it gives its chosen letter ranks answers well: AUROC 0.92–0.98.
  - From full to redacted text, stated confidence drops by 0.000 while accuracy drops 1.7 points for neckline and 2.3 for sleeve.
- **The errors are concentrated, not spread.**
  - Neckline: 98.9% on T-shirts, 72.2% on Tops. Scoop neck was right 3 of 8 times and Mandarin collar 0 of 7, with round versus scoop the main confusion.
  - Sleeve length: 99.4% on T-shirts, 96.4% on Tops, 78.0% on Shirts. 54 of 212 long-sleeve items were called three-quarter, and all 8 inspected were long-sleeve shirts photographed with the sleeves rolled up. That is the roll-up convention the hand check found, in photos whose text does not say so.
- **Rough held-out proxy:** reweighting dev accuracy per label to the held-out label mix gives about 80% for neckline (Boat neck has no dev items) and 93% for sleeve length. No test item was used.
- **Image budget:** a photo costs 167–194 tokens (median 194), and the longest prompt with its photo is 765 tokens. The budget is frozen at photos of at most 512 px on the long side.
- **Text-only redaction check:**
  - The confident text-only answers that beat the category guess rely on styling. For polo collars it is "buttoned chest placket" (all 34 such neckline items). For sleeveless it is how women's tops are described (cowl or ruffled necks).
  - Two statements survived: "vest top" and a lone "three quarter" with no "sleeves" after it. They are candidates for word list v1.1.

**Decision:** the plan's stop rule (base model above 90% on redacted dev) fired. The author chose to continue with the current task rather than add a harder attribute or switch to a smaller model, for the reasons in the plan's decision log. Two follow-ups:
- GRPO will train only on items where the model's samples disagree.
- The report now shows accuracy by category: with full text, Tops reach 84.4% on neckline and Shirts 85.6% on sleeve length, against 72.2% and 78.0% redacted.

The two text-only flags ("vest", a lone "three quarter") turned out to be loose correlates in train and dev, so word list v1 stays frozen and these dev results stand.
