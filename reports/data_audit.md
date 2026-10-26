# Data audit: Myntra Fashion Product Images, ontology v0

Parsed 44,446 records (0 unreadable); source fingerprint `f1352f4f7be4`, ontology `v0` (`1b76185bf82f`). 15,712 products are in the 3 seen and 8 held-out categories.

## Coverage by category

| Category | Side | Products | neckline values | sleeve_length values |
|---|---|---|---|---|
| Tshirts | seen | 7070 | 6065 | 6270 |
| Tops | seen | 1762 | 1555 | 1734 |
| Shirts | seen | 3217 | 0 | 2957 |
| Kurtas | held-out | 1844 | 561 | 624 |
| Kurtis | held-out | 234 | 69 | 79 |
| Tunics | held-out | 229 | 94 | 104 |
| Dresses | held-out | 464 | 390 | 220 |
| Nightdress | held-out | 189 | 161 | 82 |
| Night suits | held-out | 141 | 137 | 138 |
| Sweatshirts | held-out | 285 | 85 | 271 |
| Sweaters | held-out | 277 | 25 | 27 |

## Neckline

### Decisions

| Outcome | Seen | Held-out |
|---|---|---|
| included | 6829 | 1236 |
| text_conflict | 413 | 108 |
| family_disagreement | 158 | 30 |
| roll_up_sleeves | 0 | 0 |
| dropped_label | 175 | 89 |
| unmapped_value | 0 | 0 |
| crossing_family | 16 | 4 |
| no_usable_photo | 8 | 1 |
| unsupported_label | 21 | 54 |

### Included labels by split

| Label | train | dev | cal_fit | cal_threshold | test | heldout_test |
|---|---|---|---|---|---|---|
| Round neck | 3304 | 503 | 232 | 310 | 509 | 707 |
| V-neck | 237 | 43 | 18 | 18 | 29 | 336 |
| Scoop neck | 69 | 8 | 7 | 7 | 4 | 2 |
| Boat neck | 31 | 0 | 5 | 0 | 4 | 33 |
| Square neck | 15 | 0 | 0 | 0 | 0 | 0 |
| Mandarin collar | 45 | 7 | 1 | 7 | 5 | 156 |
| Polo collar | 959 | 148 | 67 | 68 | 112 | 0 |
| Shirt collar | 16 | 0 | 0 | 0 | 0 | 0 |
| Cowl neck | 30 | 7 | 2 | 0 | 2 | 2 |

### Label-noise estimate

Share of mapped labels whose listing text names a different label. These are excluded from training and the clean test sets.

| Category | Mapped labels | Text names another label | Rate |
|---|---|---|---|
| Kurtis | 69 | 18 | 26% |
| Tops | 1380 | 337 | 24% |
| Night suits | 137 | 28 | 20% |
| Dresses | 311 | 31 | 10% |
| Nightdress | 156 | 8 | 5% |
| Sweatshirts | 80 | 4 | 5% |
| Kurtas | 561 | 24 | 4% |
| Tunics | 90 | 3 | 3% |
| Tshirts | 6049 | 141 | 2% |
| Sweaters | 25 | 0 | 0% |

### Included labels: does the text name the answer?

| Side | Included | Text names the label | Text names no label |
|---|---|---|---|
| seen | 6829 | 85% | 15% |
| held-out | 1236 | 78% | 22% |

### Guessing the most common label per category (no photo, no text)

| Split | Included | Accuracy |
|---|---|---|
| test | 665 | 77% |
| heldout_test | 1236 | 61% |

### Redaction check (evaluation splits)

- Redacted texts that still contain an attribute word or label phrase: 0 of 3,359.
- Descriptions that become empty: 0 (0%).
- Median share of description length kept: 95%.
- The stronger check, a text-only model on redacted dev text scoring near the category guess, runs in Milestone 2.

## Sleeve length

### Decisions

| Outcome | Seen | Held-out |
|---|---|---|
| included | 10069 | 1424 |
| text_conflict | 234 | 67 |
| family_disagreement | 71 | 16 |
| roll_up_sleeves | 559 | 31 |
| dropped_label | 0 | 0 |
| unmapped_value | 0 | 0 |
| crossing_family | 16 | 6 |
| no_usable_photo | 12 | 1 |
| unsupported_label | 0 | 0 |

### Included labels by split

| Label | train | dev | cal_fit | cal_threshold | test | heldout_test |
|---|---|---|---|---|---|---|
| Short sleeves | 4977 | 771 | 348 | 426 | 716 | 472 |
| Three-quarter sleeves | 93 | 14 | 10 | 4 | 9 | 280 |
| Long sleeves | 1410 | 215 | 96 | 97 | 220 | 321 |
| Sleeveless | 463 | 65 | 31 | 32 | 72 | 351 |

### Label-noise estimate

Share of mapped labels whose listing text names a different label. These are excluded from training and the clean test sets.

| Category | Mapped labels | Text names another label | Rate |
|---|---|---|---|
| Dresses | 220 | 24 | 11% |
| Kurtis | 79 | 8 | 10% |
| Nightdress | 82 | 7 | 9% |
| Tops | 1725 | 127 | 7% |
| Night suits | 138 | 9 | 7% |
| Shirts | 2957 | 105 | 4% |
| Kurtas | 624 | 21 | 3% |
| Tunics | 103 | 2 | 2% |
| Tshirts | 6263 | 66 | 1% |
| Sweatshirts | 266 | 1 | 0% |
| Sweaters | 27 | 0 | 0% |

### Included labels: does the text name the answer?

| Side | Included | Text names the label | Text names no label |
|---|---|---|---|
| seen | 10069 | 73% | 27% |
| held-out | 1424 | 79% | 21% |

### Guessing the most common label per category (no photo, no text)

| Split | Included | Accuracy |
|---|---|---|
| test | 1017 | 85% |
| heldout_test | 1424 | 61% |

### Redaction check (evaluation splits)

- Redacted texts that still contain an attribute word or label phrase: 0 of 4,550.
- Descriptions that become empty: 0 (0%).
- Median share of description length kept: 97%.
- The stronger check, a text-only model on redacted dev text scoring near the category guess, runs in Milestone 2.

## Colour-variant families

- 12,452 families cover 15,712 in-scope products; 5,595 products (36%) share a family.
- Family sizes (5 means 5 or more): 1: 10,117, 2: 1,824, 3: 319, 4: 113, 5: 79
- Families spanning seen and held-out categories, excluded: 12.

