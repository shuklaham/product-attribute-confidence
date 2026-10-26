# Project plan: calibrated confidence under category shift

Status: Milestone 2 runs done (2026-10-26). The base model scores 95.5% (neckline) and 93.6% (sleeve length) on redacted dev, above the 90% stop line, while its stated confidence carries no information (0.95 on nearly every answer) and its letter probabilities do. How to proceed is under decision; see `docs/log.md`. Last updated 2026-10-26.

## 1. Question

When a vision-language model is post-trained to state its confidence, does reinforcement learning with a calibration reward make that confidence more reliable on product categories it never trained on, compared with supervised fine-tuning followed by cheap post-hoc calibration?

The practical use is selective auto-fill: fill a product attribute only when the model is confident enough, and leave it blank otherwise. Correct means agreement with the catalog label, which is not independently verified truth.

### Prior work this builds on

- **RLCR** ([arXiv:2507.16806](https://arxiv.org/abs/2507.16806)): adding a Brier-score term to a correctness reward improves the calibration of text-only reasoning models without hurting accuracy.
- **VL-Calibration** ([arXiv:2604.09529](https://arxiv.org/abs/2604.09529)): GRPO with a Brier-style confidence reward on Qwen3-VL-4B-Instruct, with RLCR as the strongest baseline. It reports accuracy, AUROC and ECE on standard benchmarks. It does not compare against post-hoc calibration, report risk–coverage, control the distribution shift, or remove text evidence.
- **CalibSFT** ([arXiv:2609.32470](https://arxiv.org/abs/2609.32470)): stated confidence concentrates on a few high values, RL alone does not fix this, and SFT on measured success rates helps.
- **DualRead** ([arXiv:2609.06419](https://arxiv.org/abs/2609.06419)): training answers and confidence jointly under GRPO may push confidence toward near-binary values.
- **Kalai et al.** ([arXiv:2509.04664](https://arxiv.org/abs/2509.04664)): accuracy-only grading rewards guessing over expressing uncertainty.

### What this project adds

1. A controlled shift: held-out product categories that share the label set with the training categories.
2. Evidence removal: a text-redacted condition that makes the image carry the answer, with checks that the redaction worked.
3. A fair cheap baseline: SFT plus post-hoc calibration, and answer-token probabilities with temperature scaling.
4. Decision-level metrics: risk–coverage curves and auto-fill coverage at a fixed error target.

## 2. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-12 | Use the Myntra Fashion Product Images dataset | Neckline and sleeve length are structured fields separate from the description; several categories share the labels; MIT license on Kaggle |
| 2026-10-12 | Qwen3-VL-4B-Instruct, LoRA on the language model, vision encoder frozen | Fits one 48 GB GPU; same base model as VL-Calibration |
| 2026-10-18 | Compute on Runpod Secure Cloud, A40 48 GB, with a network volume | Cheapest 48 GB card in stock that can attach persistent storage |
| 2026-10-20 | Run TRL GRPO rollouts without vLLM | TRL issue [#6294](https://github.com/huggingface/trl/issues/6294) (open at the time) corrupts multimodal prompts under vLLM; outputs are about 20 tokens, so vLLM saves little. Re-check before Milestone 4 |
| 2026-10-24 | Exclude products whose description names a different label than the catalog, from training and from the clean test set; report the rate as a label-noise estimate | Training on contradictions teaches the model to distrust correct text |
| 2026-10-24 | Limit neckline labels to mutually exclusive shapes | Strap and shoulder styles are a different concept and co-occur with neckline shapes |
| 2026-10-25 | Build the manifests from the official Kaggle records | They are byte-identical to the mirror used for exploration (source fingerprint `f1352f4f7be4`) |
| 2026-10-25 | A product needs a photo that decodes, with a long side of at least 480 px | 3 photos are missing and 10 are 150×200 px, too small to show a neckline |
| 2026-10-25 | Redaction removes statements of the attribute, not every correlated word; the text-only score on redacted text becomes the reference line the photo must beat | After two rounds of the CPU leak check, the remaining clue words are brands, fabrics and styling. Removing those would strip legitimate product copy. Word list v0 is frozen |
| 2026-10-25 | Draw the hand-check sample of 100 contradictions from training-side splits only, and freeze it (`annotations/hand_check_queue_v0.csv`) | Inspecting test items could influence decisions about them. The sample is fixed so later matcher changes do not reshuffle it |
| 2026-10-25 | Claude reviews the full sample (verdicts marked `reviewer = claude`); the author checks them in the review app, and the agreement rate is reported | A model judging necklines has unknown error, so its verdicts are labeled and checked against a person's. Superseded below: the first check was not blind |
| 2026-10-25 | Fix four phrase patterns found by the hand check: "three quarter" must lead to "sleeve"; drop "spread collar", "armholes" and "mid-arm"; count "deep round neck" as scoop | 5 of the 100 sampled contradictions were matcher misreads; 95 were genuine |
| 2026-10-25 | Metrics count invalid outputs as wrong, never auto-fill them, and rank them last in the risk–coverage area; tied confidences are averaged over a random order | An invalid output can never be auto-filled. Ranked last, it scores like a wrong answer with the lowest confidence, which can still beat a confidently wrong answer, so the invalid rate is reported beside every ranking metric. Stated confidences pile up on a few values, so any fixed order within ties would be arbitrary |
| 2026-10-25 | The auto-fill threshold is the lowest one whose accepted items on the threshold half of the calibration split have at most 5% observed error, with no correction for choosing among many thresholds | The seen test shows the resulting overshoot without any shift, so the held-out minus seen difference isolates the shift. A rule based on an upper confidence bound would guarantee the in-distribution error but cut coverage sharply on about 400 items |
| 2026-10-25 | Read the log-probabilities of all 12 letter tokens at the position where the model writes its choice (vLLM `logprob_token_ids`); pin the model revision (`ebb281ec`) and vLLM 0.32.0 in `uv.lock` | Every offered letter gets an exact probability, with no top-k cutoff. The tokenizer makes the letter a token of its own in every output layout checked |
| 2026-10-25 | Exclude products whose text describes roll-up sleeves from the sleeve-length task, in every split (`exclude_if` in the ontology) | Their length depends on a convention, since they are usually photographed rolled. Roll-up sleeves came up in 15 of the 40 genuine sleeve contradictions in the hand check, 14 of them undecidable from the photo. The text rule caught 8 of those 14, and roll-ups visible only in the photo stay as label noise |
| 2026-10-25 | Re-freeze the data as v1 after an independent review, before any training. Families also join products with byte-identical photos | The catalog lists some garments twice under different IDs with no colour link. In v0, 246 such photo groups crossed splits, and 6–7% of seen-test items had an identical photo in training. In v1, no photo appears in two splits; 291 products changed split |
| 2026-10-25 | Redaction v1: drop every clause that names a label, contains a cue or mentions the attribute; add abbreviations (SS, HS, LS, FS, S/L, Sless), misspellings, and redaction-only cues (armhole, strap); in titles, remove the words that describe a mention | Rule 3 had dropped a clause only when an attribute word was left, so "has adjustable straps" became "has adjustable". With the misspellings, brand abbreviations, and the cue "armholes", which the hand-check fix had removed from redaction along with matching, a statement or cue survived in 5.3% of held-out redacted sleeve items, against 1.3% in the seen test. Word list v1 is frozen |
| 2026-10-25 | Judge whether the text names a label on the title and description only, the text the model sees | The model never sees the variant name; with the v1 word list, including it would change 223 checks |
| 2026-10-25 | A family whose variants include a kept and a dropped value (such as round neck and halter neck) disagrees | Such families describe one garment two ways |
| 2026-10-25 | The roll-up rule needs "sleeve" or "cuff" near the phrase, and the catalog field Sleeve Styling = Roll-Up Sleeves also excludes | 30 exclusions had no "sleeve" or "cuff" near the phrase ("denims rolled up to the ankle"), and the catalog field marked 9 included items as roll-ups |
| 2026-10-25 | Any (product, attribute) a reviewer has looked at that lands in a test split is excluded from every test set | 8 items reviewed earlier are in test splits under v1: the 7 opened in the app's Browse mode and one hand-check item moved there by the photo families. The app no longer shows test splits |
| 2026-10-25 | Build the test items excluded only as label noise (contradictions and family disagreements) now, marked `clean = false`, for the "all labeled items" results | Building them before any test evaluation means the test data need not be touched again |
| 2026-10-25 | Remove runs of garbled characters from titles and descriptions | 37 products had them; 16 prompts exceeded the model context, and the longest description was 27,000 characters of them. The longest prompt is now 608 text tokens |
| 2026-10-25 | The author's 18 of 20 spot-check agreement is reported as a confirmation rate, an upper bound: the app showed Claude's verdict and pre-selected it. The app now hides Claude's verdict until the reviewer saves, and blind agreement is measured on unreviewed items | Agreement measured with the answer on screen is anchored |
| 2026-10-25 | The primary comparison uses the Brier score with invalid outputs counted as loss 1, pooled over both attributes. M2's Platt map is refitted in every bootstrap draw on a resample of the calibration families | That Brier variant cannot be improved by invalid outputs. Refitting includes the uncertainty of fitting the map, which treating it as fixed would leave out |
| 2026-10-25 | Before the Platt fit, stated confidence is clipped to [0.005, 0.995], so a stated 1.0 reads as half a step above 0.99 | The clip had been 0.0001, which set where a cluster of stated 1.0s sits on the logit axis and moved the fit; it is now fixed in advance, with a sensitivity check in the Milestone 3 report |
| 2026-10-26 | Run inference pods without a network volume, in whichever data center has A40 stock; copy results back and delete the pod | EU-SE-1 had A40 stock but offers no network volumes, and CA-MTL-1, which does, had little stock. The data is 530 MB and setup takes about 5 minutes, so a volume is not worth tying every pod to one data center. Storage for training is decided in Milestone 3 |
| 2026-10-26 | Freeze the image budget: one photo per product, at most 512 px on the long side | Measured 167–194 image tokens per photo on dev; the longest prompt with its photo is 765 tokens |

## 3. Data

**Source.** Myntra "Fashion Product Images Dataset" on Kaggle (`paramaggarwal/fashion-product-images-dataset`): one photo and one JSON record per product. Of the in-scope photos, 11,060 are 1800×2400 and 4,562 are 1080×1440. The exploratory audit (`reports/data_audit.md`) read the JSONs from a GitHub mirror. Frozen manifests use the official download, with file hashes recorded.

**Labels (ontology v1).** Counts are included labels in the train / held-out test splits, after the exclusions below and before the training cap (`manifests/v1`).

| Attribute | Kept labels | Dropped labels |
|---|---|---|
| Neckline | Round 3,301 / 707 · Polo collar 952 / 0 · V-neck 234 / 334 · Scoop 70 / 2 · Mandarin collar 45 / 156 · Boat 31 / 33 · Cowl 29 / 0 · Shirt collar 17 / 0 · Square 14 / 0 | Strap and shoulder styles: Shoulder straps, Strapless, Off-shoulder, One shoulder, Halter. Rare or not shared: Tie-up, High neck, Mock neck, Keyhole, Henley, Hood |
| Sleeve length | Short 4,984 / 472 · Long 1,400 / 321 · Sleeveless 470 / 351 · Three-quarter 90 / 280 | None. "No Sleeves" merges into Sleeveless |

Every kept label gets a one-line definition, which is shown to the model with the options. Square neck, Shirt collar and Cowl neck have fewer than 30 training examples, so they stay as options but are never scored in any evaluation split.

The label mix shifts along with the categories. Mandarin collar is 1.0% of training necklines but 12.7% of held-out ones, and three-quarter sleeves are 1.3% of training sleeves but 19.7% of held-out ones (1.4% and 2.0% of training after the cap). Calibration under this shift therefore has to cope with changed label frequencies as well as unfamiliar products.

**Splits.** Assigned by family, so all colour variants of a product, and all listings with a byte-identical photo, share a split.

| Split | Categories | Use |
|---|---|---|
| Train (70%) | T-shirts, Tops, Shirts (sleeve only) | SFT and RL |
| Dev (10%) | same | Prompt, settings, checkpoint choice, redaction word list |
| Calibration (10%) | same | Half fits confidence maps, half picks thresholds |
| Seen test (10%) | same | Final within-category evaluation |
| Held-out test | Kurtas, Kurtis, Tunics, Dresses, Nightdress, Night suits, Sweatshirts, Sweaters | Final transfer evaluation only |

**Exclusions.**
- The title or description names a different label than the catalog. By category this is 2–26% of neckline labels, highest in Kurtis (26%) and Tops (24%), against 2% for T-shirts and 4% for Kurtas. For sleeve length it is 0–11%, highest in Dresses. These rates are the label-noise estimate reported with the results.
- Labels outside ontology v1.
- The 12 families that span seen and held-out categories.
- Families whose variants disagree on a label, counting dropped values (90 neckline, 45 sleeve): drop that attribute for the family.
- Labels with fewer than 30 training examples are never scored, in any evaluation split.
- Products without a usable photo: it must decode and have a long side of at least 480 px.
- Any item a reviewer has looked at that falls in a test split (8 items).
- Test items excluded only as label noise (contradictions and family disagreements) are still built, marked `clean = false`: 101 in the seen test and 197 in the held-out test.

**Balancing.** Cap each (category, label) cell in training at 1,500 examples to start; 95% of T-shirt sleeve labels are Short. The cap shifts the training label mix away from the natural mix of the calibration and test splits (Round neck 70% before the cap, 58% after; Short sleeves 72% and 56%). RL calibrates its confidence on the training mix while the Platt map is fitted on the natural one, so the RL data mix is decided before Milestone 4.

**Images.** One photo per product (the default view), resized to at most 512 px on the long side; 46 photos stay at 480 px, because smaller photos are not enlarged. That gives at most 256 image tokens per product. The image token budget is fixed in Milestone 2 and not changed afterwards.

## 4. Evidence conditions and redaction

Every evaluation item is scored under two conditions, reported separately.

| Condition | Model input | Role |
|---|---|---|
| Full text | Title, cleaned description, photo | Practical extraction; secondary |
| Redacted | Title and description with label phrases removed, photo | Primary condition for every claim about the image or about calibration |

The variant name and URLs are never part of the input; both can contain the label.

**Redaction rules.**
1. The redaction step never sees the label, so its output cannot depend on the answer.
2. Remove the phrases and synonyms of every label of the attribute, not only the correct one, including abbreviations and misspellings, plus redaction-only cues that point to a value without naming one (armhole, strap).
3. Remove every clause that names a label, contains a cue or mentions the attribute, whole ("has adjustable straps"), so no gap or remnant shows what was removed. The title and the opening noun phrase keep the product type and lose only the mention and the words that describe it ("Three Fourth Sleeve Kurta" becomes "Kurta").
4. Keep brand names intact (for example, "U.S. Polo Assn.").

**Checks.**
- No attribute word, label phrase or cue may survive redaction outside a brand name in any evaluation item. The audit checks every item (`reports/data_audit.md`); 56 neckline evaluation items keep "Polo" inside a brand name (U.S. Polo Assn., Classic Polo), as rule 4 intends. An independent scan with a broader list of misspellings and abbreviations found no statements left either.
- A text-only classifier trained on redacted train text and scored on dev must not rely on statements of the attribute. `scripts/leak_check.py` runs a naive Bayes version on CPU; the model-based version runs in Milestone 2. Clue words that state the attribute go into the word list. Brand, fabric and styling correlates stay, and the text-only score is reported as the reference line the photo must beat. The word list is frozen before any test evaluation: v0 on 2026-10-25, and v1, after the review, on the same day.
- Products whose text names no label (15–27% of included labels) form a naturally silent group. Comparing results on that group with the redacted group, within each category, shows whether redaction itself distorts results.

**Training mix.** A seeded 50/50 draw of full or redacted text for each (family, attribute) presentation, identical across all models.

**Ablations.** On a fixed subset of 500 test items per attribute, evaluation only: photo only, text only (full and redacted), and photo swap (the text paired with another product's photo that has a different label).

## 5. Task format

The prompt gives the instructions, the evidence, the attribute definition, and every kept label as a lettered option with its definition. Letters are assigned by a seeded permutation that is fixed per item across models and reshuffled during training.

The model returns `{"choice": "<letter>", "confidence": <number from 0 to 1>}`. Parsing is strict: malformed JSON, extra keys, unknown letters and out-of-range confidence are invalid, count as wrong, and are reported. Evaluation uses greedy decoding; RL rollouts sample.

## 6. Conditions

The core study compares five conditions.

| ID | Condition | Updated parameters |
|---|---|---|
| M0 | Qwen3-VL-4B-Instruct, prompted | None |
| M1 | LoRA SFT on correct answers | Language-model LoRA |
| M2 | M1 plus a Platt map on stated confidence | Two scalars |
| M3 | M1 plus GRPO, reward = correctness | Language-model LoRA |
| M4 | M1 plus GRPO, reward = correctness − (confidence − correctness)² | Language-model LoRA |

Two more comparators are reported separately:
- **M2-logit** (Milestone 3): M1's answer-letter probabilities with temperature scaling. Inference only.
- **M5** (Milestone 5): SFT on measured per-item success rates, sampled from M1, in the style of CalibSFT.

**SFT.** LoRA rank 16 on the language model's attention and MLP projections, with target modules named explicitly so the vision encoder stays untouched. The learning rate is picked from {5e-5, 1e-4, 2e-4} on dev. The loss covers the answer letter and the JSON syntax; the confidence value is masked out of the loss, because the catalog has no true confidence to copy. Format validity is checked on dev before the protocol is frozen.

**GRPO.**
- 8 samples per prompt, temperature 1.0, at most 32 completion tokens, about 500 steps of 8 prompts.
- Invalid output gets a reward of −1.1 in both arms.
- M3 and M4 share every setting except the reward: starting checkpoint, data order, seeds, steps, KL coefficient and advantage normalization.
- Logged per run: correctness and calibration terms separately, confidence histograms, the share of groups where all 8 rewards are equal (no learning signal), KL, gradient norms and throughput.

## 7. Evaluation

**Test grid.** {seen test, held-out test} × {full, redacted}, on the clean subset and on all labeled items. All labeled items adds the test items excluded only as label noise, contradictions and family disagreements: 101 in the seen test and 197 in the held-out test.

**Metrics.**
- Accuracy, pooled and averaged over category–attribute groups.
- Brier score, on valid outputs and with invalid outputs counted as loss 1.
- ECE with 10 bins, plus reliability diagrams.
- AUROC of confidence for correctness.
- Risk–coverage curve and its area.
- Coverage and observed error at a threshold chosen on the threshold half of the calibration split (target: at most 5% error), then applied unchanged.
- Evidence sensitivity: the drop in confidence from full to redacted text compared with the drop in accuracy, both on the items valid in both conditions, with the change in the invalid rate reported beside them.
- Invalid-output rate next to every ranking metric (ECE, AUROC, risk–coverage), since an invalid output ranks like a wrong answer with the lowest confidence; GPU-hours and cost.

**Statistics.** Differences between methods get 95% intervals from a paired bootstrap over product families (10,000 resamples). For M2, the Platt map is refitted in every draw on a resample of the calibration families, so its intervals include the uncertainty of fitting the map. Variation across training seeds is reported separately.

**Primary comparison, fixed before any training.** M4 against M2 on the held-out test, redacted text, clean subset, pooled over both attributes. M4 counts as better if the 95% interval of the difference in the Brier score, with invalid outputs counted as loss 1, lies below zero, and the lower bound of the accuracy difference (invalid outputs counted as wrong) is above −1 point. Secondary comparisons: M4 against M3 (the effect of the reward) and M4 against M2-logit.

All of these outcomes are reportable: RL wins under shift; the cheap fixes match it; every method degrades under shift; or confidence ignores the missing evidence.

## 8. Compute and budget

- **GPU:** Runpod Secure Cloud A40 48 GB, $0.59/hr (catalog read 2026-10-18), in CA-MTL-1.
- **Storage:** a 30 GB high-performance network volume in the same data center, $0.14/GB-month.
- **Software on the pod:** `uv sync --group gpu` installs vLLM 0.32.0 with torch 2.13.0. Those torch wheels use CUDA 13, so the pod's host needs NVIDIA driver 580 or newer, and pods are created with that CUDA filter.
- **Estimate:** 70–100 GPU-hours, about $55–75 including storage. Spending past $100 requires a review.
- **Tracking:** spend is read from Runpod billing at each milestone and recorded in `docs/log.md`.
- **Local machine:** data preparation, tests and analysis.

## 9. Milestones

Each milestone ends in something runnable or readable, and the core is delivered before the deeper extensions.

| # | Milestone | Exit criteria |
|---|---|---|
| 1 | **Data foundation:** official download and images verified, ontology v0, tightened matcher with a 100-item hand check, frozen family splits, redaction v0, example builder, tests | Manifests (IDs and hashes) committed; tests pass for no family overlap, no held-out leakage, no label in the prompt, and alias mapping; audit v2 |
| 2 | **End-to-end baseline:** Runpod pod and volume, vLLM inference with log-probabilities, strict parser, metrics module, M0 on dev in both conditions, text-only redaction check | M0 dev table; redaction check passes; invalid-output rate known and fixed if high; image budget frozen |
| 3 | **Supervised result:** M1, M2 and M2-logit, full test evaluation of M0–M2 | Interim report `reports/phase1.md` with reliability and risk–coverage figures; M1's confidence shown to vary (not constant) |
| 4 | **RL result:** GRPO smoke tests (reward unit tests, decoded rollouts), M3 and M4 with one seed each, full evaluation | Five-condition results table; zero-signal group share logged for each arm |
| 5 | **Depth:** three seeds for M3 and M4, M5, photo-swap test, 300 held-out items hand-labeled from photos blind to the catalog label, final write-up | Final report with intervals, seed spread and failure analysis |

**Stop and rethink if:**
- the model-based text-only check finds direct statements of the attribute that the word list misses, and two rounds of additions do not remove them;
- M0 on redacted dev is above 90% (saturated) or below 40% (the photo carries too little signal);
- more than 80% of GRPO groups give no learning signal (adjust on dev only);
- spend passes $100.

## 10. Risks

| Risk | Signal | Mitigation |
|---|---|---|
| Label noise differs by category and confounds the shift result | Neckline conflict rates of 0–26% by category | Clean subset is primary; report noise per category; 300 hand-labeled items |
| Sleeve-length labels are ambiguous for roll-up sleeves and cap sleeves | In the hand check, 15 of the 40 genuine sleeve contradictions involved roll-up sleeves, mostly photographed rolled, and 7 involved cap sleeves versus sleeveless | Roll-up sleeves described in the text are excluded. Report per-label reliability, so the cap versus sleeveless boundary and undetected roll-ups show up as label effects |
| Neighbouring neckline labels are hard to tell apart | Round versus scoop and round versus boat dominate neckline contradictions. Both disagreements in the hand-check spot check were round versus boat | Report per-label reliability and the confusion between neighbouring labels; treat errors on these boundaries as partly label ambiguity |
| The label mix shifts between seen and held-out categories | Mandarin collar and three-quarter sleeves are rare in training but common in held-out categories | Report per-label results and reliability by label; check how much of any calibration gap comes from label frequency |
| Leakage through text | Text-only redacted accuracy well above the category guess | Redaction rules and checks above; naturally silent control group |
| Leakage across splits | Same family, or the same photo, in two splits | Family-level splits that also join byte-identical photos, exclusion of crossing families, tests. Near-duplicate photos (re-shot or re-encoded) are not caught |
| Task saturation | M0 near ceiling on redacted dev | Change the label set or attributes before Milestone 3 |
| SFT breaks the output format | High invalid-output rate on dev | Masked-confidence SFT target, fixed in Milestone 2 |
| Confidence collapses to a few values | Narrow confidence histograms | Track histograms; M5 as an alternative starting point |
| GRPO gets little signal | High zero-signal group share | Dev-only adjustment of sampling temperature and data mix |
| Tooling bugs | Garbled decoded rollouts | Decode-and-check test before every RL run |
| Overrun | Spend or schedule past plan | Milestone exit criteria; the supervised result stands alone |

## 11. Engineering standards

- Layout: `src/` code, `configs/` settings, `manifests/` item IDs and hashes (no raw data), `reports/` results, `docs/` plan and log.
- Each run writes a run record: git commit, config hash, data manifest hash, model revision, package versions, seed, GPU type, GPU-hours and cost.
- Tests run with `uv run pytest`; CI runs them on every push.
- One branch and pull request per milestone.
- `docs/log.md` gets a dated entry per working session: what ran, what it cost, what it showed, and what changed.

## 12. Out of scope for now

Training the vision layers (a possible ablation), more attributes, a second catalog, extra photo views, larger models, human preference data, and LLM judges.
