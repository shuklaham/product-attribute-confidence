# Data audit: Myntra Fashion Product Images, ontology v1

Parsed 44,446 records (0 unreadable); source fingerprint `f1352f4f7be4`, ontology `v1` (`a3179ca26da6`). 15,712 products are in the 3 seen and 8 held-out categories.

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
| included | 6806 | 1232 |
| text_conflict | 403 | 107 |
| family_disagreement | 180 | 30 |
| roll_up_sleeves | 0 | 0 |
| dropped_label | 175 | 89 |
| unmapped_value | 0 | 0 |
| crossing_family | 16 | 4 |
| no_usable_photo | 8 | 1 |
| unsupported_label | 32 | 56 |
| inspected_test_item | 0 | 3 |

### Included labels by split

| Label | train | dev | cal_fit | cal_threshold | test | heldout_test |
|---|---|---|---|---|---|---|
| Round neck | 3301 | 503 | 228 | 309 | 509 | 707 |
| V-neck | 234 | 44 | 18 | 17 | 28 | 334 |
| Scoop neck | 70 | 8 | 8 | 7 | 4 | 2 |
| Boat neck | 31 | 0 | 5 | 0 | 4 | 33 |
| Square neck | 14 | 0 | 0 | 0 | 0 | 0 |
| Mandarin collar | 45 | 7 | 1 | 7 | 5 | 156 |
| Polo collar | 952 | 148 | 67 | 69 | 117 | 0 |
| Shirt collar | 17 | 0 | 0 | 0 | 0 | 0 |
| Cowl neck | 29 | 0 | 0 | 0 | 0 | 0 |

### Label-noise estimate

Share of mapped labels whose listing text names a different label. These are excluded from training and the clean test sets.

| Category | Mapped labels | Text names another label | Rate |
|---|---|---|---|
| Kurtis | 69 | 18 | 26% |
| Tops | 1380 | 333 | 24% |
| Night suits | 137 | 28 | 20% |
| Dresses | 311 | 31 | 10% |
| Sweaters | 25 | 2 | 8% |
| Nightdress | 156 | 8 | 5% |
| Sweatshirts | 80 | 4 | 5% |
| Kurtas | 561 | 24 | 4% |
| Tunics | 90 | 3 | 3% |
| Tshirts | 6049 | 144 | 2% |

### Included labels: does the text name the answer?

| Side | Included | Text names the label | Text names no label |
|---|---|---|---|
| seen | 6806 | 85% | 15% |
| held-out | 1232 | 78% | 22% |

### Guessing the most common training label of the category (no photo, no text)

Held-out categories have no training labels, so they get the most common training label overall.

| Split | Included | Accuracy |
|---|---|---|
| dev | 710 | 71% |
| test | 667 | 76% |
| heldout_test | 1232 | 57% |

### Redaction check (evaluation splits)

- Redacted texts that still contain an attribute word, label phrase or redaction cue outside the brand name: 0 of 3,345.
- Redacted texts whose only such word is inside the brand name, kept by rule 4 (for example U.S. Polo Assn.): 56.
- Descriptions that become empty: 0 (0%).
- Median share of description length kept: 92%.
- The stronger check, a text-only model on redacted dev text scoring near the category guess, runs in Milestone 2.

## Sleeve length

### Decisions

| Outcome | Seen | Held-out |
|---|---|---|
| included | 10073 | 1424 |
| text_conflict | 210 | 64 |
| family_disagreement | 92 | 16 |
| roll_up_sleeves | 556 | 31 |
| dropped_label | 0 | 0 |
| unmapped_value | 0 | 0 |
| crossing_family | 16 | 6 |
| no_usable_photo | 12 | 1 |
| unsupported_label | 0 | 0 |
| inspected_test_item | 2 | 3 |

### Included labels by split

| Label | train | dev | cal_fit | cal_threshold | test | heldout_test |
|---|---|---|---|---|---|---|
| Short sleeves | 4984 | 777 | 349 | 434 | 711 | 472 |
| Three-quarter sleeves | 90 | 14 | 10 | 4 | 9 | 280 |
| Long sleeves | 1400 | 212 | 94 | 96 | 231 | 321 |
| Sleeveless | 470 | 60 | 29 | 30 | 69 | 351 |

### Label-noise estimate

Share of mapped labels whose listing text names a different label. These are excluded from training and the clean test sets.

| Category | Mapped labels | Text names another label | Rate |
|---|---|---|---|
| Dresses | 220 | 24 | 11% |
| Kurtis | 79 | 8 | 10% |
| Nightdress | 82 | 7 | 9% |
| Tops | 1725 | 127 | 7% |
| Night suits | 138 | 9 | 7% |
| Shirts | 2957 | 106 | 4% |
| Kurtas | 624 | 21 | 3% |
| Tunics | 103 | 2 | 2% |
| Tshirts | 6263 | 66 | 1% |
| Sweatshirts | 266 | 1 | 0% |
| Sweaters | 27 | 0 | 0% |

### Included labels: does the text name the answer?

| Side | Included | Text names the label | Text names no label |
|---|---|---|---|
| seen | 10073 | 73% | 27% |
| held-out | 1424 | 79% | 21% |

### Guessing the most common training label of the category (no photo, no text)

Held-out categories have no training labels, so they get the most common training label overall.

| Split | Included | Accuracy |
|---|---|---|
| dev | 1063 | 86% |
| test | 1020 | 87% |
| heldout_test | 1424 | 33% |

### Redaction check (evaluation splits)

- Redacted texts that still contain an attribute word, label phrase or redaction cue outside the brand name: 0 of 4,553.
- Redacted texts whose only such word is inside the brand name, kept by rule 4 (for example U.S. Polo Assn.): 0.
- Descriptions that become empty: 0 (0%).
- Median share of description length kept: 96%.
- The stronger check, a text-only model on redacted dev text scoring near the category guess, runs in Milestone 2.

## Families

Families join colour variants and, with photos.csv, products whose photo files are byte-identical.

- 11,881 families cover 15,712 in-scope products; 6,614 products (42%) share a family.
- Family sizes (5 means 5 or more): 1: 9,098, 2: 2,167, 3: 413, 4: 121, 5: 82
- Families spanning seen and held-out categories, excluded: 12.

