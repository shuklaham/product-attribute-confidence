# Project plan: calibrated confidence under category shift

Status: Milestone 1 (data foundation) complete. The hand check is done: Claude reviewed 100 contradictions, and the author agreed on 18 of 20 in a spot check. Milestone 2 is next. Last updated 2026-10-25.

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
| 2026-10-25 | Claude reviews the full sample (verdicts marked `reviewer = claude`); the author confirms them in the review app, and the agreement rate is reported | A model judging necklines has unknown error, so its verdicts are labeled and checked against a person's |
| 2026-10-25 | Fix four phrase patterns found by the hand check: "three quarter" must lead to "sleeve"; drop "spread collar", "armholes" and "mid-arm"; count "deep round neck" as scoop | 5 of the 100 sampled contradictions were matcher misreads; 95 were genuine |
| 2026-10-25 | Exclude products whose text describes roll-up sleeves from the sleeve-length task, in every split (`exclude_if` in the ontology) | Their length depends on a convention, since they are usually photographed rolled. They were 14 of 43 sleeve contradictions in the hand check. The rule removes 590 sleeve labels; it catches 8 of those 14, and roll-ups visible only in the photo stay as label noise |

## 3. Data

**Source.** Myntra "Fashion Product Images Dataset" on Kaggle (`paramaggarwal/fashion-product-images-dataset`): one 1800×2400 photo and one JSON record per product. The exploratory audit (`reports/data_audit.md`) read the JSONs from a GitHub mirror. Frozen manifests use the official download, with file hashes recorded.

**Labels (ontology v0).** Counts are included labels in the train / held-out test splits, after the exclusions below (`manifests/v0`).

| Attribute | Kept labels | Dropped labels |
|---|---|---|
| Neckline | Round 3,304 / 705 · Polo collar 955 / 0 · V-neck 237 / 336 · Mandarin collar 45 / 156 · Scoop 66 / 2 · Boat 31 / 33 · Cowl 30 / 2 · Square 15 / 0 · Shirt collar 21 / 0 | Strap and shoulder styles: Shoulder straps, Strapless, Off-shoulder, One shoulder, Halter. Rare or not shared: Tie-up, High neck, Mock neck, Keyhole, Henley, Hood |
| Sleeve length | Short 5,028 / 483 · Long 1,701 / 321 · Sleeveless 466 / 354 · Three-quarter 109 / 291 | None. "No Sleeves" merges into Sleeveless |

Every kept label gets a one-line definition, which is shown to the model with the options. Square neck and Shirt collar have fewer than 30 training examples, so they stay as options but are never scored.

The label mix shifts along with the categories. Mandarin collar is 1% of training necklines but 13% of held-out ones, and three-quarter sleeves are 1.5% of training sleeves but 20% of held-out ones. Calibration under this shift therefore has to cope with changed label frequencies as well as unfamiliar products.

**Splits.** Assigned by colour-variant family, so all variants of a product share a split.

| Split | Categories | Use |
|---|---|---|
| Train (70%) | T-shirts, Tops, Shirts (sleeve only) | SFT and RL |
| Dev (10%) | same | Prompt, settings, checkpoint choice, redaction word list |
| Calibration (10%) | same | Half fits confidence maps, half picks thresholds |
| Seen test (10%) | same | Final within-category evaluation |
| Held-out test | Kurtas, Kurtis, Tunics, Dresses, Nightdress, Night suits, Sweatshirts, Sweaters | Final transfer evaluation only |

**Exclusions.**
- The description names a different label than the catalog. By category this is 0–26% of neckline labels, highest in Kurtis (26%) and Tops (25%), against 2% for T-shirts and 4% for Kurtas. For sleeve length it is 0–11%, highest in Dresses. These rates are the label-noise estimate reported with the results.
- Labels outside ontology v0.
- The 12 families that span seen and held-out categories.
- Families whose variants disagree on a label (86 neckline, 40 sleeve): drop that attribute for the family.
- Held-out labels with fewer than 30 training examples stay out of the primary transfer test.
- Products without a usable photo: it must decode and have a long side of at least 480 px.

**Balancing.** Cap each (category, label) cell in training at 1,500 examples to start; 95% of T-shirt sleeve labels are Short.

**Images.** One photo per product (the default view), resized to 512 px on the long side. The image token budget is fixed in Milestone 2 and not changed afterwards.

## 4. Evidence conditions and redaction

Every evaluation item is scored under two conditions, reported separately.

| Condition | Model input | Role |
|---|---|---|
| Full text | Title, cleaned description, photo | Practical extraction; secondary |
| Redacted | Title and description with label phrases removed, photo | Primary condition for every claim about the image or about calibration |

The variant name and URLs are never part of the input; both can contain the label.

**Redaction rules.**
1. The redaction step never sees the label, so its output cannot depend on the answer.
2. Remove the phrases and synonyms of every label of the attribute, not only the correct one.
3. Remove whole clauses ("has a round neck"), so no gap shows what was removed.
4. Keep brand names intact (for example, "U.S. Polo Assn.").

**Checks.**
- No attribute word or label phrase may survive redaction in any evaluation item. The audit checks every item (`reports/data_audit.md`).
- A text-only classifier trained on redacted train text and scored on dev must not rely on statements of the attribute. `scripts/leak_check.py` runs a naive Bayes version on CPU; the model-based version runs in Milestone 2. Clue words that state the attribute go into the word list. Brand, fabric and styling correlates stay, and the text-only score is reported as the reference line the photo must beat. The word list is frozen before any test evaluation; v0 was frozen on 2026-10-25.
- Products whose description names no label (14–26% of included labels) form a naturally silent group. Comparing results on that group with the redacted group, within each category, shows whether redaction itself distorts results.

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

**Test grid.** {seen test, held-out test} × {full, redacted}, on the clean subset and on all labeled items.

**Metrics.**
- Accuracy, pooled and averaged over category–attribute groups.
- Brier score, on valid outputs and with invalid outputs counted as loss 1.
- ECE with 10 bins, plus reliability diagrams.
- AUROC of confidence for correctness.
- Risk–coverage curve and its area.
- Coverage and observed error at a threshold chosen on the threshold half of the calibration split (target: at most 5% error), then applied unchanged.
- Evidence sensitivity: per item, the drop in confidence from full to redacted text compared with the drop in accuracy.
- Invalid-output rate, GPU-hours and cost.

**Statistics.** Differences between methods get 95% intervals from a paired bootstrap over product families (10,000 resamples). Variation across training seeds is reported separately.

**Primary comparison, fixed before any training.** M4 against M2 on the held-out test, redacted text, clean subset. M4 counts as better if the 95% interval of the Brier difference lies below zero and the lower bound of the accuracy difference is above −1 point. Secondary comparisons: M4 against M3 (the effect of the reward) and M4 against M2-logit.

All of these outcomes are reportable: RL wins under shift; the cheap fixes match it; every method degrades under shift; or confidence ignores the missing evidence.

## 8. Compute and budget

- **GPU:** Runpod Secure Cloud A40 48 GB, $0.59/hr (catalog read 2026-10-18), in CA-MTL-1.
- **Storage:** a 30 GB high-performance network volume in the same data center, $0.14/GB-month.
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
| Sleeve-length labels are ambiguous for roll-up sleeves and cap sleeves | In the hand check, 14 of 43 sleeve contradictions were roll-up sleeves photographed rolled, and 4 were cap versus sleeveless | Roll-up sleeves described in the text are excluded. Report per-label reliability, so the cap versus sleeveless boundary and undetected roll-ups show up as label effects |
| Neighbouring neckline labels are hard to tell apart | Round versus scoop and round versus boat dominate neckline contradictions. Both disagreements in the hand-check spot check were round versus boat | Report per-label reliability and the confusion between neighbouring labels; treat errors on these boundaries as partly label ambiguity |
| The label mix shifts between seen and held-out categories | Mandarin collar and three-quarter sleeves are rare in training but common in held-out categories | Report per-label results and reliability by label; check how much of any calibration gap comes from label frequency |
| Leakage through text | Text-only redacted accuracy well above the category guess | Redaction rules and checks above; naturally silent control group |
| Leakage across splits | Same family in two splits | Family-level splits, exclusion of crossing families, tests |
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
