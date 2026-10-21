# Data audit: Myntra Fashion Product Images

Parsed 44,446 product records (0 unreadable). 15,712 are in the 3 seen and 8 held-out categories.

## Label coverage by category

| Category | Split | Products | Neckline labels | Sleeve labels | Extra views listed |
|---|---|---|---|---|---|
| Tshirts | seen | 7070 | 6065 | 6270 | 100% |
| Tops | seen | 1762 | 1555 | 1734 | 100% |
| Shirts | seen | 3217 | 0 | 2957 | 100% |
| Kurtas | held-out | 1844 | 561 | 624 | 100% |
| Kurtis | held-out | 234 | 69 | 79 | 100% |
| Tunics | held-out | 229 | 94 | 104 | 100% |
| Dresses | held-out | 464 | 390 | 220 | 100% |
| Nightdress | held-out | 189 | 161 | 82 | 100% |
| Night suits | held-out | 141 | 137 | 138 | 100% |
| Sweatshirts | held-out | 285 | 85 | 271 | 100% |
| Sweaters | held-out | 277 | 25 | 27 | 100% |

## Neckline

### Label counts

| Label | Seen categories | Held-out categories | Note |
|---|---|---|---|
| Round Neck | 5085 | 792 |  |
| Polo Collar | 1378 | 0 |  |
| V-Neck | 400 | 353 |  |
| Scoop Neck | 277 | 6 |  |
| Mandarin Collar | 79 | 163 |  |
| Boat Neck | 79 | 41 |  |
| Square Neck | 51 | 44 |  |
| Shirt Collar | 49 | 32 |  |
| Shoulder Straps | 75 | 3 |  |
| Strapless | 26 | 51 | too little training support |
| Cowl Neck | 47 | 2 |  |
| Tie-Up Neck | 30 | 0 |  |
| Off-Shoulder | 2 | 20 | too little training support |
| Halter Neck | 6 | 11 | too little training support |
| High Neck | 16 | 0 |  |
| One Shoulder | 10 | 0 |  |
| Keyhole Neck | 7 | 0 |  |
| Henley Neck | 3 | 0 |  |
| Mock Neck | 0 | 2 | too little training support |
| Hood | 0 | 2 | too little training support |

### Guessing the most common label per category

| Split | Labeled products | Accuracy |
|---|---|---|
| seen | 7620 | 67% |
| held-out | 1522 | 55% |

### How often the text gives the label away

| Split | N | Attribute word in title | in variant name | in description | Text names the label | Text names another label | Text names none |
|---|---|---|---|---|---|---|---|
| seen | 7620 | 1% | 2% | 89% | 79% | 8% | 14% |
| held-out | 1522 | 0% | 6% | 89% | 69% | 14% | 17% |

"Text names another label" marks possible label noise or a phrase-matching miss; inspect samples before excluding.

## Sleeve length

### Label counts

| Label | Seen categories | Held-out categories | Note |
|---|---|---|---|
| Short Sleeves | 7543 | 533 |  |
| Long Sleeves | 2524 | 336 |  |
| Sleeveless | 695 | 370 |  |
| Three-Quarter Sleeves | 199 | 306 |  |

### Guessing the most common label per category

| Split | Labeled products | Accuracy |
|---|---|---|
| seen | 10961 | 84% |
| held-out | 1545 | 60% |

### How often the text gives the label away

| Split | N | Attribute word in title | in variant name | in description | Text names the label | Text names another label | Text names none |
|---|---|---|---|---|---|---|---|
| seen | 10961 | 0% | 1% | 82% | 71% | 3% | 26% |
| held-out | 1545 | 1% | 5% | 91% | 72% | 5% | 23% |

"Text names another label" marks possible label noise or a phrase-matching miss; inspect samples before excluding.

## Colour-variant families

- 12,452 families cover 15,712 in-scope products; 5,595 products (36%) share a family with another product.
- Family sizes (5 means 5 or more): 1: 10,117, 2: 1,824, 3: 319, 4: 113, 5: 79
- Families spanning seen and held-out categories: 12 (keep together or exclude).
- Families with conflicting labels: neckline: 86, sleeve_length: 40

