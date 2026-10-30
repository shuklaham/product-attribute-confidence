# Project plan: calibrated confidence under category shift

Status: Milestone 6 (three seeds of recipe v1 and of M5, evidence runs) complete; results in `reports/phase1.md` to `reports/phase4.md`, and the study is summarized in `docs/writeup.md`.
- **Primary comparison, recipe v1 over three seeds (Milestone 6):** M4 does not count as better than M2-logit on the held-out test with redacted text: mean Brier difference +0.004 (−0.003 to +0.010). Milestone 4's near-miss (−0.006) was the best of the three seeds. Across seeds, M4 beats M3 (Brier −0.012) and M1 (+0.9 points of accuracy), and ranks answers worse than M2-logit (AUROC −0.065).
- **Recipe v2 over three seeds (Milestone 5):** M4 v2 is worse than M2-logit (+0.017, +0.009 to +0.025). With almost no wrong answers left in training, the Brier reward inflated the written confidence.
- **M5 (CalibSFT-style SFT):** its written confidence ranks answers best among written confidences in every seed (AUROC 0.833–0.836) but is miscalibrated (Brier 0.095–0.105).
- **The test of M5-logit (Milestone 6):** not confirmed. On two new seeds the mean Brier difference against M2-logit is −0.005 (−0.011 to +0.002), though M5-logit has the lowest Brier score of any method in every seed.
- **Evidence:** with neither the photo nor the label in the text, accuracy falls to about 52%, while the written confidence of the SFT and RL models stays at 0.78–0.87. M2-logit's letter probability falls to 0.60, closest to the accuracy. When the photo contradicts the text, M5's written confidence and M2-logit's letter probability fall about 20 points, M4's about 5.

Last updated 2026-10-30.

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
| 2026-10-26 | Continue with both attributes and the held-out primary comparison, although the base model scores above the 90% stop line on redacted dev (author's decision) | The difficulty is concentrated rather than absent: Tops score 72% on neckline, scoop necks and Mandarin collars are mostly missed, and long-sleeve shirts photographed rolled up are called three-quarter. Reweighted to the held-out label mix, neckline accuracy would be about 80%. The study's phenomenon is clear: stated confidence is 0.95 on nearly every answer (AUROC 0.50) while the letter probabilities rank answers well (AUROC 0.92–0.98). Results are reported by category so T-shirts cannot hide the hard slices |
| 2026-10-26 | GRPO trains only on training items where the model's sampled answers are neither all right nor all wrong; M3 and M4 use the same pool | A group whose 8 answers are all right, or all wrong, gives every sample the same correctness reward and no learning signal. With T-shirts near 99%, an unfiltered pool would be mostly such groups |
| 2026-10-26 | The primary comparison becomes M4 against M2-logit; M4 against M2 becomes secondary; M1 stays plain SFT (author's decision, before any training) | The base model states 0.95 on nearly every answer, and SFT with the confidence value left out of the loss will most likely keep that habit, so a Platt map on M1's stated confidence would turn a constant into another constant. Temperature-scaled letter probabilities are the strongest cheap fix: on the base model they rank answers with AUROC 0.92–0.98. The question becomes whether RL can make the stated confidence as informative as what the model already holds internally, under shift |
| 2026-10-26 | Keep word list v1. "Vest" and a lone "three quarter", which the model-based check flagged, are not added | In train and dev, "vest" appears in 22 redacted sleeve items and only 9 are sleeveless; a lone "three quarter" mostly describes trousers (1 of 5 is three-quarter sleeves). They are correlates, not statements |
| 2026-10-28 | GPU pods stop themselves. The run script stops its pod when it finishes and after a time limit; outputs are written to the pod's volume disk, which a stop keeps; the laptop checks progress with short SSH calls that retry, then copies the outputs back and deletes the pod | The Milestone 3 pod finished its work in 4 hours and then billed 33 more hours idle (about $19.50 of the $21.74 spent so far). The only watcher was one SSH connection, which dropped with the laptop's network. A stopped pod bills only its volume disk |
| 2026-10-28 | GRPO samples from two strata, half of each batch from each: prompts whose 8 sampled M1 answers disagree (mixed) and the rest (settled). This replaces "only mixed prompts" | On dev, M1's letter probabilities predict that only 5.5% of groups at temperature 1 hold both right and wrong answers (`scripts/signal_share.py`). An unfiltered pool would give the correctness reward no signal in about 95% of groups, past the 80% stop line. A mixed-only pool fixes that for M3 but would never show M4 an easy item, so its Brier term would learn when to doubt but not when to be sure, the wrong lesson for the roughly 90% of held-out answers that are right. Half and half gives M3 signal in at least half the groups and keeps both lessons for M4 |
| 2026-10-28 | Sample at temperature 1 | The model's own distribution. A higher temperature would enlarge the mixed stratum (21% at 1.5, 46% at 2.0 on dev) by training on answers the model would not give, and add format errors |
| 2026-10-28 | Advantage = reward minus the group mean, with no division by the group's standard deviation (as in Dr. GRPO); no KL term | In a group of right answers, M4's rewards differ only through the stated confidence, by thousandths. Dividing by their tiny spread would make those differences full-size updates, as large as a right-versus-wrong gap. The unscaled advantage keeps every update proportional to its reward difference. With one update per batch on fresh samples and a short run, the KL term mainly costs a reference forward pass (about a quarter of the compute); recent GRPO recipes drop it too |
| 2026-10-28 | Steps: 3 passes over the mixed stratum, at most 500; learning rate 1e-5, untuned | The mixed stratum is small, about 425 prompts if training items behave like dev. Each pass draws fresh samples, but more passes over the same few hundred prompts invite overfitting. The learning rate is an order of magnitude below the SFT rates, the usual gap for policy gradients, which are noisier; there is no budget to tune it |
| 2026-10-28 | Our own GRPO loop rather than TRL's trainer | TRL's multimodal path had the vLLM bug (#6294), and it has not been shown to pass Qwen3-VL's image-token markers under transformers 5.19, which SFT needed. The loop is about 300 lines: it samples with the model's own `generate`, and a check compares its training log-probabilities with the sampler's. It also skips groups whose advantages are all zero exactly |
| 2026-10-28 | Train and evaluate M4 before M3 | M4 is in the primary comparison; if the run is cut short, that result is ready first |
| 2026-10-28 | Milestone 5 retrains M3 and M4 with GRPO recipe v2, three seeds each (author's decision after the 80% stop line was crossed). Every 50 steps the current policy scores every prompt by p, the probability it gives the right letter where it writes its choice (one forward pass). A prompt is mixed if 8 samples would hold both right and wrong answers with probability 1 − p⁸ − (1 − p)⁸ ≥ 0.2. Everything else stays as in v1, including 461 steps and half of each batch from each stratum. If one full scoring pass costs more than 6 minutes in the pre-flight check, re-scoring runs every 100 steps instead | Under v1 the strata were fixed at the start, and the share of groups with no correctness signal rose from 73% to 97% as the policy grew sure, even on prompts it had not yet trained on, so the policy's current uncertainty must be measured as it changes. One forward pass gives the exact chance a group is mixed, where 8 samples give a noisy count. On M1's dev predictions the threshold makes 5.6% of prompts mixed, at a 63% chance of a mixed group. The cost rule is set on cost, before any result |
| 2026-10-28 | The Milestone 5 primary comparison is the mean over seeds 0–2 of the M4 v2 minus M2-logit Brier difference (held-out test, redacted text, clean items, pooled). Its 95% interval comes from a paired family bootstrap that resamples the test families once per draw for all seeds and refits M2-logit every draw. M4 counts as better if the interval lies below zero and the lower bound of the mean accuracy difference is above −1 point. Each seed is also reported alone. The v1 runs stay Milestone 4's result | One training seed cannot separate the method from the run. Averaging over seeds and keeping the same decision rule makes the result comparable with Milestone 4's |
| 2026-10-28 | M5 continues M1's adapter with CalibSFT-style SFT (arXiv:2609.32470). The data is M1's own 8 sampled answers per RL prompt, from the v1 sampling pass. Each answer's confidence target is 0.5 q + 0.5 z, with q the prompt's success rate and z the answer's correctness. At most 300 answers per target value are kept (3,458; 53% wrong). The loss is 0.5 × the mean loss on confidence tokens (all answers) plus 0.5 × the mean loss on the other answer tokens (right answers only). One epoch, learning rate 1e-4, batch 16, cosine; no RL after it | CalibSFT reports what this project found, stated confidence crowding onto a few high values, and that SFT on measured success rates helps. Its recipe balances answers equally across target values. Here the rarest value has 30 answers, so a cap of 300 keeps enough data while balancing. The cap leans the training mix toward hard prompts, unlike the test mix; M5 is evaluated as trained, and that is reported |
| 2026-10-28 | The photo swap uses 500 clean held-out items per attribute, in both text conditions. Each item gets a donor photo: a clean product of the same category, from another family, with a different label. Answers are scored against the text's label and the photo's label, and confidence is compared with the same items unswapped, for M0, M1, M2-logit, each M3 and M4 seed, and M5 | It tests whether confidence notices conflicting evidence. 76 neckline items have no donor (no same-category product with another label) and are left out |
| 2026-10-29 | Milestone 6 reruns GRPO recipe v1, unchanged, with seeds 1 and 2 for M4 and M3 (author's decision). The seed sets the schedule (seed 1000 × seed) and the sampling draws (seed 1,000,003 × seed + step), so seed 0 is the Milestone 4 run. A pre-flight on the pod reruns the first 3 steps of seed 0 and compares them with the Milestone 4 log, since the trainer was reorganized for v2 | Three seeds were planned for Milestone 5 whatever the outcome. The Milestone 4 interval crossed zero by 0.0004, which is exactly what one seed cannot settle, and v2 did not replace v1 as the better recipe |
| 2026-10-29 | The Milestone 6 primary comparison is the mean over seeds 0–2 of the M4 v1 minus M2-logit Brier difference, with the Milestone 5 bootstrap and decision rule: M4 counts as better if the interval lies below zero and the lower bound of the mean accuracy difference is above −1 point. Each seed is also reported alone. M4 v1 against M3 v1, averaged over the seeds, is the secondary comparison | The same statistic and rule as Milestone 5, so the two recipes are compared on equal terms. Seed 0's result is known, and it stays in the mean because it was planned as one of the three |
| 2026-10-29 | The exploratory M5-logit result is tested on new training runs: M5 is retrained with seeds 1 and 2 (same data, loss and steps; the seed sets the data order and the dropout). The test is the mean over seeds 1 and 2 of the M5-logit minus M2-logit Brier difference, each temperature refitted on every draw, with the same rule. Seed 0 suggested the hypothesis, so it is reported but left out of the test. M5's written confidence is reported per seed, without a test | A result found by looking at the data cannot be confirmed on the same run. New seeds test whether the gain belongs to the method or to one training run. They are scored on the same held-out test set, so they cannot show that this set does not happen to favour M5-logit; the bootstrap covers the sampling of test families, and two post-hoc variants of M5 were tried before M5-logit was singled out |
| 2026-10-29 | On the photo-swap subset, unswapped, M0, M1, M4 v1 seed 0 and M5 seed 0 run once with only the photo (text removed; one text condition, since without the text the two are the same input) and once with only the text (both conditions). Reported descriptively: accuracy and stated confidence per evidence mode, beside the same items with both and swapped. No decision rule | The swap showed that most models' written confidence ignores contradicting evidence. Removing one source at a time shows how much each model's answers and confidence rest on each, on the same items as the swap |

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

The core study compares five model conditions, M0 to M4, plus M2-logit, an inference-only baseline built on M1.

| ID | Condition | Updated parameters |
|---|---|---|
| M0 | Qwen3-VL-4B-Instruct, prompted | None |
| M1 | LoRA SFT on correct answers | Language-model LoRA |
| M2 | M1 plus a Platt map on stated confidence | Two scalars |
| M2-logit | M1's answer-letter probabilities with temperature scaling, the main cheap baseline | One scalar |
| M3 | M1 plus GRPO, reward = correctness | Language-model LoRA |
| M4 | M1 plus GRPO, reward = correctness − (confidence − correctness)² | Language-model LoRA |

M2-logit is inference only: it reads the probability M1 gives each offered letter where it writes its choice, and rescales those probabilities with one temperature fitted on the calibration split. It answers with M1's letter, so its accuracy is M1's.

One more comparator is reported separately: **M5** (Milestone 5), SFT on measured per-item success rates, sampled from M1, in the style of CalibSFT. Its recipe is in the decision log (2026-10-28) and in `src/pac/calibsft.py`.

**SFT.** LoRA rank 16 on the language model's attention and MLP projections, with target modules named explicitly so the vision encoder stays untouched. The learning rate is picked from {5e-5, 1e-4, 2e-4} on dev. The loss covers the answer letter and the JSON syntax; the confidence value is masked out of the loss, because the catalog has no true confidence to copy. Format validity is checked on dev before the protocol is frozen.

**GRPO** (settled 2026-10-28, before any RL run; reasons in the decision log).
- **Prompts:** each training item once, as SFT's second pass presents it (its seeded condition and option order).
- **Strata:** before RL, M1 answers every prompt 8 times at temperature 1 (`scripts/sample_pool.py`). A prompt is *mixed* if those answers are partly right and partly wrong, and *settled* otherwise.
- **Batches:** each step takes 4 mixed and 4 settled prompts, in a seeded order that both arms share.
- **Sampling:** 8 samples per prompt at temperature 1, from the full distribution (no top-k or top-p), with at most 48 completion tokens, the evaluation budget.
- **Steps:** enough for 3 passes over the mixed prompts (3 × mixed / 4), at most 500.
- **Advantage:** a sample's reward minus its group's mean, not divided by the group's standard deviation. Groups whose advantages are all zero are skipped.
- **No KL term:** one update per batch, on samples from the same policy.
- **Optimizer:** the LoRA starts as M1's adapter, with dropout off. AdamW at learning rate 1e-5 after 10 warm-up steps, betas (0.9, 0.99), no weight decay, gradients clipped at 1.
- **Invalid output** gets a reward of −1.1 in both arms.
- **Shared settings:** M3 and M4 share everything except the reward: starting checkpoint, prompts, data order, sampling seeds, steps and optimizer. M4 runs first.
- **Logged every step:** reward, accuracy, invalid rate, stated confidence overall and for right and wrong answers, the stated-value histogram, the share of groups with no correctness signal, groups trained, gradient norm, time and memory.
- **Before the long runs:** a check on a few prompts prints decoded rollouts with their rewards in both arms. It also confirms that sampling used the plain distribution, and that the training forward scores sampled tokens as the sampler did.
- **Recipe v2 (Milestone 5):** the strata are not fixed. Every 50 steps the current policy scores every prompt by its probability p of the right letter, and a prompt is mixed if 1 − p⁸ − (1 − p)⁸ ≥ 0.2. Seeds 0, 1 and 2 set the schedule and the sampling draws. The check also confirms that every sampled answer opens with the prefix the scorer assumes, and that the scorer's p matches the sampler's.

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

**Statistics.** Differences between methods get 95% intervals from a paired bootstrap over product families (10,000 resamples). For M2 and M2-logit, the Platt map and the temperature are refitted in every draw on a resample of the calibration families, so their intervals include the uncertainty of fitting the map. Variation across training seeds is reported separately.

**Primary comparison, fixed before any training.** M4 against M2-logit on the held-out test, redacted text, clean subset, pooled over both attributes. M4 counts as better if the 95% interval of the difference in the Brier score, with invalid outputs counted as loss 1, lies below zero, and the lower bound of the accuracy difference (invalid outputs counted as wrong) is above −1 point. Secondary comparisons: M4 against M3 (the effect of the reward) and M4 against M2 (the Platt map on stated confidence). Until 2026-10-26 the primary comparison was M4 against M2; the decision log gives the reason for the change, made before any training. In Milestone 5 the same comparison is made with M4 recipe v2, averaged over three training seeds (decision log, 2026-10-28), and in Milestone 6 with recipe v1 over three seeds (decision log, 2026-10-29).

All of these outcomes are reportable: RL wins under shift; the cheap fixes match it; every method degrades under shift; or confidence ignores the missing evidence.

## 8. Compute and budget

- **GPU:** Runpod Secure Cloud A40 48 GB, $0.59/hr (catalog read 2026-10-18), in whichever data center has stock.
- **Storage:** no network volume (decision log, 2026-10-26). Each pod writes its outputs to its own volume disk; they are copied back, checked against checksums, and the pod is deleted.
- **Software on the pod:** `uv sync --group gpu` installs vLLM 0.32.0 with torch 2.13.0. Those torch wheels use CUDA 13, so the pod's host needs NVIDIA driver 580 or newer, and pods are created with that CUDA filter.
- **Estimate:** 70–100 GPU-hours, about $55–75 including storage. Spending past $100 requires a review.
- **Tracking:** spend is read from Runpod billing at each milestone and recorded in `docs/log.md`.
- **Spent:** $58.34 through Milestone 6 (billing read 2026-10-30): $21.74 through Milestone 3, of which about $19.50 was a pod left idle after its work finished; about $4.78 for Milestone 4; $20.76 for Milestone 5 and $10.52 for Milestone 6 (per-pod billing). The other $0.54 is usage that posted after earlier readings.
- **Local machine:** data preparation, tests and analysis.

## 9. Milestones

Each milestone ends in something runnable or readable, and the core is delivered before the deeper extensions.

| # | Milestone | Exit criteria |
|---|---|---|
| 1 | **Data foundation:** official download and images verified, ontology v0, tightened matcher with a 100-item hand check, frozen family splits, redaction v0, example builder, tests | Manifests (IDs and hashes) committed; tests pass for no family overlap, no held-out leakage, no label in the prompt, and alias mapping; audit v2 |
| 2 | **End-to-end baseline:** Runpod pod and volume, vLLM inference with log-probabilities, strict parser, metrics module, M0 on dev in both conditions, text-only redaction check | M0 dev table; redaction check passes; invalid-output rate known and fixed if high; image budget frozen |
| 3 | **Supervised result:** M1, M2 and M2-logit, full test evaluation of M0–M2 | Interim report `reports/phase1.md` with reliability and risk–coverage figures, and the distribution of M1's stated confidence (the base model states 0.95 on nearly every answer) |
| 4 | **RL result:** GRPO smoke tests (reward unit tests, decoded rollouts), M3 and M4 with one seed each, full evaluation | Results table for M0–M4 and M2-logit; zero-signal group share logged for each arm |
| 5 | **Depth:** three seeds for M3 and M4, M5, photo-swap test, 300 held-out items hand-labeled from photos blind to the catalog label, final write-up | Final report with intervals, seed spread and failure analysis |
| 6 | **Seeds and evidence:** seeds 1 and 2 of recipe v1 for M3 and M4 and of M5, photo-only and text-only runs on the photo-swap subset, the write-up | `reports/phase4.md` with the three-seed verdict for M4 v1, the test of M5-logit, and the evidence table |

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
