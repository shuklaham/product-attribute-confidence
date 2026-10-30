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

**Second decision: the main comparison changes before any training.** SFT with the confidence value left out of the loss will most likely keep the base model's habit of stating 0.95. A Platt map on that would be a constant. The author chose the recommended fix:
- the primary comparison becomes M4 against M2-logit, the temperature-scaled letter probabilities, which are the strongest cheap fix;
- M4 against M2 becomes secondary;
- M1 stays plain SFT;
- Milestone 3 reports M1's confidence distribution instead of requiring it to vary.

**Milestone 2 exit criteria:**
- the dev table is in `reports/m0_dev.md`;
- the model-based redaction check found no statements beyond two loose correlates;
- the invalid-output rate is known (0%);
- the image budget is frozen.

Milestone 2 is complete.

## 2026-10-26 to 2026-10-28: Milestone 3, SFT and the cheap fixes

**Ran**
- One A40 pod (Runpod Secure Cloud, EU-SE-1) without a network volume. A 6-step training smoke test passed, then `scripts/pod/phase1.sh` ran unattended at commit `fdbc5b9` for 3 hours 45 minutes:
  - M0 on both calibration halves and both test splits;
  - LoRA SFT at learning rates 5e-5, 1e-4 and 2e-4: one pass over the 7,717 training items, 483 steps of 16 examples, 64 minutes each, each followed by the adapter on dev;
  - M1, at the chosen rate, on both calibration halves and both test splits.
- The outputs were copied back and checked against checksums made on the pod (38 files), then the pod was deleted.
- `scripts/report_phase1.py` fitted the maps and thresholds locally, and wrote `reports/phase1.md` and two figures in `reports/figures/`.

**Cost**
- Runpod billing, read after the delete: $21.74 for the project so far ($21.44 GPU, $0.30 disk). Nearly all of it is this pod: 37 hours at $0.59/hr.
- About 4 hours of that were work (about $2.40). The other 33 hours (about $19.50) were idle. The pipeline finished at 19:35 UTC on 2026-10-26, but the only thing watching it was one SSH connection from the laptop. That connection dropped when the laptop lost its network, and nothing on the pod shut it down, so it billed until the next session found it on 2026-10-28.
- The fix is in the decision log: from Milestone 4 on, the pod stops itself.
- Lost: the console output of each step (`/root/logs/` on the pod), which was not copied before the delete. The training loss curves survive in each run record (`log_history`, every 10 steps).

**Findings** (redacted text, clean items, both attributes pooled; intervals are 95% from the paired family bootstrap)
- **Learning rate.** All three rates reach 98.1–98.4% on redacted dev with no invalid outputs, and the rule picks 2e-4. The training loss ends near 0.006 for all three, because most target tokens are fixed JSON.
- **SFT raises accuracy, also on categories it never saw.**
  - Seen test: 93.8% to 98.4% (+4.6 points, +3.4 to +5.9).
  - Held-out test: 82.8% to 89.3% (+6.6 points, +5.1 to +8.2).
  - Shirts go from 76.9% to 99.6%: the catalog's convention for rolled-up sleeves is learned. Nightdress is the one held-out category that gets worse, from 92.6% to 86.8% (204 items).
- **M1's stated confidence still says almost nothing.** It states 0.95 on 89.6% of answers, 0.9 on 5.4% and 0.98 on 5.0%. Its AUROC is 0.66 on the seen test and 0.63 on the held-out test, against 0.50 for M0. On held-out items, redacting the text lowers accuracy by 7.2 points and stated confidence by 0.5 points.
- **M2-logit is the strongest method on every metric where the methods can differ.**
  - Held-out Brier score 0.078, against 0.096 for M1 (difference −0.017, −0.025 to −0.008) and 0.099 for M2 (−0.020, −0.053 to −0.009).
  - AUROC 0.86 against 0.63 for M1, pooled over both attributes. Within each attribute, which leaves out the easier-attribute effect: 0.78 (neckline) and 0.84 (sleeve length), against 0.59 and 0.55.
  - ECE 0.040 against 0.051.
  - On held-out items, its confidence falls 9.8 points when the text is redacted, while accuracy falls 7.2.
  - Its strength comes from SFT, not from the temperature. The base model's letter probabilities need temperatures of 4.4–6.4; M1's need 1.01–1.17, and on the held-out test M1's raw letter probabilities have the same Brier score, 0.078, as the temperature-scaled ones.
- **The Platt map on stated confidence does not survive the shift.** On the seen test it lowers ECE from 0.034 to 0.013. On the held-out test it raises ECE from 0.051 to 0.080 and does not improve the Brier score (+0.003, −0.005 to +0.036). On seen categories a stated 0.95 is right 97–99% of the time, and the map carries that over to categories where it is not.
- **Thresholds chosen on seen categories do not transfer.**
  - M1 is 97.4% right on the threshold half, so for M1, M2 and M2-logit the lowest qualifying threshold accepts nearly everything.
  - On held-out items, those frozen thresholds auto-fill 99.5–99.7% of answers at 10.5–10.6% error, twice the 5% target.
  - M2-logit's ranking allows more: its most confident 83.9% of held-out answers have 5% error. No threshold fitted on seen categories finds that point.
- **Clip check:** M1 states no value outside [0.005, 0.995], so M2 does not depend on the clip.
- No invalid outputs in any run.

**Decision:** GPU pods stop themselves (decision log, 2026-10-28).

**Milestone 3 exit criteria:** `reports/phase1.md` has the reliability and risk–coverage figures and the distribution of M1's stated confidence. Milestone 3 is complete.

**Next**
- Milestone 4 needs a learning signal. M1 is about 98% right on seen categories, so most groups of 8 sampled answers will be all right and teach nothing. Measure that share before any RL run, and set the data mix and sampling temperature on dev only.
- Re-check TRL issue #6294 (vLLM with multimodal prompts) before choosing how GRPO generates its samples.
- For the author, optional: the blind review in the app (73 items left).

## 2026-10-28: Milestone 4, GRPO for M3 and M4

**Pod and the self-stop checks, before any work**
- One A40 pod (Runpod Secure Cloud, CA-MTL-1, CUDA 13.2 host), with a 20 GB volume disk at `/workspace` for every output.
- Before anything was copied to it, the self-stop path was tested on the live pod.
  - The key Runpod puts in the pod is refused by `runpodctl` 1.14 ("Unauthorized") and by the REST API (403), but accepted by the GraphQL API for that pod.
  - The first version of `scripts/pod/selfstop.sh` used `runpodctl`, so it would have failed exactly when it was needed. It now calls GraphQL.
  - A `podStop` sent from inside the pod moved it to `EXITED`. It was started again, and a file written to `/workspace` before the stop was still there.
- The whole safety chain then ran once end to end, with only the final stop as a dry run.
  - The guard saw no file change for 30 seconds, recorded why, and waited.
  - The laptop watcher saw the end, copied the outputs, checked them against checksums made on the pod, and marked them copied, within 70 seconds.
  - The guard then stopped at once instead of waiting out its full copy window.
- Pushes send only git-tracked files and the git history, so `.env` and `private/` cannot reach a pod; both were confirmed absent on it.

**Ran**
- Sampling pass (`scripts/sample_pool.py`, 14 minutes): M1 answered the 7,717 RL prompts 8 times each at temperature 1.
  - 614 prompts were mixed (8.0%; dev had predicted 5.5%) and 7,103 settled (7,053 all right, 50 all wrong).
  - Mixed prompts by group: neckline redacted 290, neckline full 129, sleeve redacted 131, sleeve full 64.
  - 96.9% of the samples were right, and 15 of 61,736 were invalid. The step rule gave 461 steps.
- Pre-flight check on 8 prompts:
  - Sampling used the plain distribution.
  - The training forward reproduced the sampler's token log-probabilities to a mean gap of 0.003 nats (max 0.12, bf16 rounding).
  - Sampling then moved to one left-padded batch per step, about 30% faster, and passed the same check.
- Smoke run: 4 M4 steps, 39 s per step, 27 GB peak memory.
- `scripts/pod/phase2.sh` at commit `1c3ec71`, 06:05–13:15 UTC. M4: 461 steps in 285 minutes, then its five evaluations. M3: 461 steps in 113 minutes, then its five. Every step exited 0.

**A pipeline bug, caught by an independent review of the running code**
- In `phase2.sh`, `run()` set `name` without `local`, which overwrote the loop's own `name`. Every evaluation after training would have looked for its adapter under the previous step's name (`artifacts/train_m4/adapter`, then `artifacts/train_m4_dev/adapter`, and so on). The first evaluation would have failed and ended the run: M4 never evaluated, M3 never trained.
- The running loop could not be edited, since bash had already parsed it. While M4 was still training, links were put at the 10 paths the loop would ask for, pointing to the real adapters. After the copy, each oddly named output folder was renamed only after its run record's split and adapter hash were checked.
- Fixed in `4bc84f1`. A plan-only mode runs the real loop with echoed commands, and `tests/test_pod_scripts.py` checks every path; it fails when the bug is put back.
- The review found nothing wrong in the training code: loss, masking, batching, rewards and arm parity.

**The pod stopped itself, and was then deleted**
- The watcher copied the outputs and checked them against checksums made on the pod at 13:17 UTC, and the pod was `EXITED` at the next read.
- Restarting it to fetch the console logs failed twice, because another renter had taken the GPU on its host. So it was deleted, and the console output in `/workspace/logs` was lost.
- Every prediction, run record and per-step training log had already been copied and verified. The watcher's timeline is in `artifacts/logs/phase2_pod/watch.log`.

**Cost:** Runpod billing, read after the delete: $26.52 for the project ($26.14 GPU, $0.39 disk), so Milestone 4 cost about $4.78 (one A40 pod, 05:24–13:17 UTC).

**Findings** (held-out test, redacted text, clean items, both attributes pooled, one training seed; intervals are 95% from the paired family bootstrap, 10,000 draws)
- **Primary comparison: M4 does not count as better than M2-logit.**
  - Brier score 0.072 against 0.078: difference −0.0061, interval −0.0136 to +0.0004. The interval crosses zero by 0.0004, so the criterion fails.
  - Accuracy +2.0 points (+1.1 to +2.8), within the allowed margin.
- **M4 is the best-calibrated method under the shift, but worse at ranking.**
  - ECE 0.015, against 0.040 for M2-logit and 0.051 for M1.
  - When M4 writes 0.85 it is right 83.7% of the time; 0.92, 92.1%; 0.98, 98.7%.
  - AUROC 0.789 against 0.862 (−0.073, −0.096 to −0.048). Within each attribute: 0.726 and 0.750, against 0.781 and 0.838.
- **Part of M4's Brier gain is accuracy.** RL raised held-out accuracy: M1 89.3%, M3 90.1%, M4 91.3%. M4 minus M3 is +1.2 points (+0.5 to +1.9); M3 minus M1 is +0.75 (+0.15 to +1.4).
- **The calibration reward matters.**
  - Against M3, M4 is better on Brier (−0.011, −0.016 to −0.007) and AUROC (+0.088, +0.062 to +0.115).
  - M3's written confidence is overconfident: its 0.8 answers are right 63% of the time.
- **The written number now follows the model's own letter probability.** Rank correlation 0.87 for M4, 0.62 for M3 and 0.51 for M1 (exploratory).
- **It now reacts to missing evidence, partly.** Redacting the text lowers M4's confidence by 4.0 points while its accuracy falls 5.8. For M1 the numbers were 0.5 and 7.2.
- **M4 writes few values.** 7 distinct; 0.98 on 45% of answers, 0.95 on 22%, below 0.9 on 28%. (Corrected 2026-10-29: this line first gave 75%, 11% and 12%, which pool both test sets and both text conditions.) A number with so few levels cannot order answers as finely as a probability, which likely explains its lower AUROC.
- **Thresholds still do not transfer.** M4's frozen thresholds auto-fill 99.8% of held-out answers at 8.6% error (M2-logit: 10.5%), still above the 5% target.
- **On the seen test the methods tie on Brier:** M4 minus M2-logit is −0.0007 (−0.0036 to +0.0021).
- **Training used up its own signal.**
  - The share of groups whose 8 answers were all right or all wrong rose from 73% in steps 0–24 to 90% in steps 100–199 and 97% after step 300. Over the run it was 91%, past the plan's 80% stop line.
  - M3 trained on 0.7 groups per step on average, so it is a weak control for "RL itself".
  - M4 trained on 7.9 groups per step, because its stated confidences still varied when its answers agreed.

**Milestone 4 exit criteria:**
- the results table for M0–M4 and M2-logit is in `reports/phase2.md`;
- the zero-signal share is logged for each arm (`artifacts/m3/log.jsonl`, `artifacts/m4/log.jsonl`).

Milestone 4 is complete.

**Next** (author's decisions for Milestone 5):
- **The stop line was crossed**, so the RL recipe needs a rethink before more seeds. Options: re-sort the pool between passes, keep only groups with signal as they are sampled, or start RL from a model that is less sure on the training categories. Whichever is chosen, it is fixed before the next run.
- **Three seeds for M3 and M4.** With one seed, the primary interval crosses zero by 0.0004, and the seed spread is unknown.

## 2026-10-28: Milestone 5, GRPO recipe v2 over three seeds, M5 and the photo swap

**Decisions** (author, before any run; reasons in the plan's decision log):
- the RL recipe is fixed (v2) and all three seeds are rerun;
- M5 and the photo swap are added;
- PRs #2–#4 were merged into main locally.

**Prepared locally**
- Photo-swap set: 500 clean held-out items per attribute, in both text conditions (2,000 examples). 1,156 of 1,232 neckline items and all 1,424 sleeve items had a donor.
- M5 data: 3,458 of M1's 61,721 valid sampled answers, capped at 300 per target value; 1,631 right and 1,827 wrong.

**Pre-flight on a live pod**
- The v2 check, with 8 prompts and seed 0:
  - sampling used the plain distribution;
  - every sampled answer opened with the prefix the scorer assumes;
  - training log-probabilities matched the sampler's (mean gap 0.0033);
  - the scorer's p matched the sampler's letter probability within 0.0023.
- A full scoring pass of the 7,717 prompts took 19.1 minutes, against an estimate of 5–11. By the rule fixed beforehand (over 6 minutes), re-scoring runs every 100 steps, not 50.
- At step 0, M1's scores made 732 prompts mixed, with a 0.63 chance of a mixed group, the same as dev predicted.
- M5: the token check marked exactly the confidence digits, and a 3-step run trained with the two-part loss.

**Ran**
- Three A40 pods (Secure Cloud; two in CA-MTL-1, one in EU-SE-1), all at commit `6388cf4`. Each runs `scripts/pod/phase3.sh` with `RESCORE=100`.
- One pod runs seed 0 then M5 and the reference swaps (`EXTRA=1`); the other two run seeds 1 and 2.
- Launched at 16:35 UTC. Each pod stops itself when its run ends, when nothing changes for 40 minutes, or after 15 hours. Three laptop watchers copy the outputs and the pods' logs.
- Expected: about 12.5 hours and about $21.50. The 19-minute scoring passes (five per arm) explain most of the rise from the $19 quoted.

**Finished.** Seeds 1 and 2 finished at 03:05 and 03:12 UTC, and seed 0 with M5 and the reference photo swaps at 03:52. Every step exited 0. Each pod stopped itself after its watcher had copied and verified its outputs, and its logs this time. All three pods were then deleted.

**Cost:** Runpod billing, read after the deletes: $47.23 for the project ($46.46 GPU, $0.77 disk), so Milestone 5 cost about $20.71.

**Findings** (held-out test, redacted text, clean items, both attributes pooled; intervals are 95% from the paired family bootstrap, 10,000 draws; full tables in `reports/phase3.md`)
- **Primary comparison: M4 v2 does not count as better than M2-logit; it is worse.** Averaged over the three seeds, the Brier difference is +0.0173 (+0.0087 to +0.0248). Per seed it is +0.017, +0.008 and +0.027.
- **Recipe v2 failed, and the mechanism is clear.**
  - Re-scoring found 732 uncertain prompts at step 0, 161–215 at step 100, 58–82 at step 200, and 4–20 after that. The policy grew certain everywhere, so re-sorting had no uncertainty left to find.
  - Each re-scoring briefly lowered the share of groups with no correctness signal, to 60–70%, but it climbed back each time and stayed at 96–99% after step 300. Over the run it was 86–88%, against 91% for v1.
  - In the last third, half of every batch came from the same 4–20 prompts, redrawn dozens of times.
  - Once the samples held almost no wrong answers, M4's Brier reward could only reward stating more confidence. Seed 0 writes 0.98 or 0.99 on every held-out answer while being right 90% of the time (ECE 0.080, AUROC 0.53). v1 kept its original hard prompts, which kept producing wrong answers, and ended at ECE 0.015.
  - M4 v2 is also worse than M3 v2 on Brier (+0.0068, +0.0041 to +0.0096).
- **M5 has the most informative written confidence of any model, but it is miscalibrated.**
  - AUROC 0.833, against 0.789 for M4 v1 and 0.631 for M1; M2-logit's letter probability has 0.862.
  - Its balanced training mix made it underconfident at the low end: answers it writes at 0–0.12 are right 52–60% of the time. That gives ECE 0.060 and Brier 0.095, worse than M2-logit (+0.0167, +0.0082 to +0.0246).
  - It is 0.7 points more accurate than M1 (+0.04 to +1.33).
- **Exploratory, found after the results:** M5's own letter probabilities, temperature-scaled ("M5-logit"), have the best held-out Brier score of any method: 0.0714, against 0.0784 for M2-logit (−0.0070, −0.0127 to −0.0015).
  - Decomposed: the same resolution, better calibration (0.0028 against 0.0051), and higher accuracy, which accounts for about three-quarters of the gain.
  - One training run and not pre-registered, so it is a hypothesis to test.
  - A Platt map on M5's written confidence does not beat M2-logit (+0.0053, −0.0034 to +0.0167).
- **Photo swap.**
  - With redacted text, every model answers with the photo's label 72–85% of the time.
  - With full text that contradicts the photo, the trained models follow the text more: M0 43%, M1 62%, M3 and M4 v2 69–74%, M5 82%.
  - The written confidence of M0, M1, M3 v2 and M4 v2 seed 0 barely moves under the conflict (0.95 or 0.98 either way).
  - M5's falls from 0.92 to 0.72, and M2-logit's letter probability from 0.95 to 0.74.

**Next** (author's decisions):
- Write up the study as it stands: the pre-registered verdicts for v1 and v2, the failure mechanism of v2, and M5's results.
- Optionally, a pre-registered follow-up of the exploratory result: M5-logit against M2-logit with three SFT seeds, about $2–3.
- Optionally, an RL recipe that keeps wrong answers in training: a KL anchor to M1, a fixed share of hard prompts in every batch, or stopping when the signal runs out.

## 2026-10-29: Milestone 6, three seeds for recipe v1, M5 seeds and the evidence runs

**Decisions** (author, before any run; reasons in the plan's decision log):
- seeds 1 and 2 of recipe v1 for M4 and M3, and of M5; the test of M5-logit leaves out seed 0, which suggested it;
- photo-only and text-only runs on the photo-swap subset;
- PR #5 was merged into main locally.

**Prepared locally**
- `train_sft.py --seed`; seed 0 is the Milestone 5 run.
- The photo-swap subset unswapped (`subset_heldout.jsonl`, 2,000 examples: the same 1,000 items in both text conditions).
- `scripts/pod/phase4.sh`, with plan-only tests of every command.

**Pre-flight on a live pod**
- Recipe v1 with seed 0 for 3 steps, against the Milestone 4 log:
  - step 0 is identical: every sampled answer, so the reward, accuracy, stated values and signal share;
  - at steps 1 and 2, accuracy and the signal share are identical and the rewards differ by at most 0.00013. That is floating-point rounding: step 0's gradient already differs in its seventh significant digit (norm 1.9926111 against 1.9926106), and the first update carries the difference forward.
- The logged learning rate is one warm-up step lower. The Milestone 4 log recorded the next step's rate, which `4bc84f1` fixed; the update itself did not change.

**Ran**
- Two A40 pods (Secure Cloud; CA-MTL-1 and EU-SE-1), at commit `b44a038`. Each runs `scripts/pod/phase4.sh` for one seed: M4 v1, then M5, then M3 v1, each evaluated on the five splits and the swap set as soon as it is trained. Then pod A (seed 1) runs the evidence runs for M0 and M1, and pod B (seed 2) for M4 and M5.
- Launched at 05:17 UTC. Each pod stops itself when its run ends, when nothing changes for 40 minutes, or after 12 hours. Two laptop watchers copy the outputs and the pods' logs.
- Expected: about 8.5 hours and $10.40 (two pods at $0.59/hr).

**Finished.** Seed 2's pod finished at 13:27 UTC and seed 1's at 13:30; every step exited 0. The laptop slept through part of the run, so the watchers' copies paused, hit a few connection resets and resumed. Both copied and verified their outputs at 13:54, inside the pods' 30-minute wait, and each pod then stopped itself. Both pods were deleted.

**Cost:** per-pod billing, read the next day: $5.26 for each pod, so Milestone 6 cost $10.52 (estimate $10.40). The project total is $58.34 ($57.36 GPU, $0.98 disk). It rose $11.11 since the last reading; the other $0.59 is earlier usage that had not posted when that total was read.

**Findings** (held-out test, redacted text, clean items, both attributes pooled; intervals are 95% from the paired family bootstrap, 10,000 draws; full tables in `reports/phase4.md`)
- **Primary comparison: M4 v1 does not count as better than M2-logit over three seeds.**
  - Mean Brier difference +0.0042 (−0.0031 to +0.0104).
  - Per seed: −0.0061 (seed 0, the Milestone 4 run), +0.0131 and +0.0057. Milestone 4's near-miss came from the best of the three seeds.
  - Accuracy +0.9 points (+0.3 to +1.5).
- **The test of M5-logit: not confirmed.** On the two new seeds the mean Brier difference is −0.0047 (−0.0111 to +0.0017); the interval crosses zero.
  - All three seeds point the same way: −0.0070, −0.0034 and −0.0061.
  - M5-logit's ECE is 0.012–0.019 in every seed, against 0.040 for M2-logit.
- **What holds across seeds:**
  - The calibration reward beats the correctness reward: M4 v1 minus M3 v1 Brier −0.0120 (−0.0151 to −0.0091), accuracy +0.4 points (+0.1 to +0.8).
  - M4's written confidence ranks answers worse than M2-logit's letter probability: AUROC −0.065 (−0.083 to −0.045).
  - M5's written confidence ranks answers the same way in every seed: AUROC 0.833–0.836, the best of any written confidence. Its ECE is 0.060–0.076.
- **RL varies a lot between seeds.**
  - M4 v1's Brier score ranges 0.072–0.091 and its ECE 0.015–0.045. Seed 0 spread its written values more (28% below 0.9, against 5–9% for seeds 1 and 2).
  - M3 v1 seeds 1 and 2 write only two values, 0.98 on 93–96% of answers (AUROC 0.52–0.54).
  - The spread of M4 v1's Brier score between seeds (0.019) is three times the difference Milestone 4 measured against M2-logit (0.006).
- **Evidence runs** (1,000 clean held-out items):
  - The photo alone gives the same accuracy and confidence as the photo with redacted text, for every model (within 1 point), so the redaction leaves little the photo does not already say.
  - The full text alone gets 91–92% right, close to photo and full text (94–96%).
  - With the redacted text alone, accuracy falls to 51–54%. Written confidence stays high for M1 (0.87), M0 (0.80) and M4 (0.78), and falls too far for M5 (0.28). M2-logit's letter probability comes closest (0.60).
- **Photo swap** (new seeds): M4 v1's written confidence falls 4–5 points when the photo contradicts the text; M3 v1's does not move (0.98). M5's falls 20–25 points in every seed, as in Milestone 5.
