# Results

- `m0_dev`, started 2026-10-26T14:12:55+00:00: Qwen/Qwen3-VL-4B-Instruct at `ebb281ec`, commit `2931c4d4`, photo_text, 3,546 items, NVIDIA A40, 2.5 min
- `m0_dev_text_only`, started 2026-10-26T14:15:41+00:00: Qwen/Qwen3-VL-4B-Instruct at `ebb281ec`, commit `2931c4d4`, text_only, 1,773 items, NVIDIA A40, 1.4 min
- `m0_dev_photo_only`, started 2026-10-26T14:17:16+00:00: Qwen/Qwen3-VL-4B-Instruct at `ebb281ec`, commit `2931c4d4`, photo_only, 1,773 items, NVIDIA A40, 1.7 min

## Answers

| Run | Split | Subset | Mode | Attribute | Text | n | Accuracy (95% interval) | Accuracy on valid outputs | Macro accuracy | Category guess | Invalid |
|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|
| m0_dev | dev | clean | photo_text | neckline | full | 710 | 97.2% (95.5% to 98.6%) | 97.2% | 91.7% | 70.8% | 0.0% |
| m0_dev | dev | clean | photo_text | neckline | redacted | 710 | 95.5% (93.6% to 97.2%) | 95.5% | 85.5% | 70.8% | 0.0% |
| m0_dev | dev | clean | photo_text | sleeve_length | full | 1,063 | 95.9% (94.5% to 97.1%) | 95.9% | 94.1% | 85.5% | 0.0% |
| m0_dev | dev | clean | photo_text | sleeve_length | redacted | 1,063 | 93.6% (91.9% to 95.2%) | 93.6% | 91.3% | 85.5% | 0.0% |
| m0_dev_photo_only | dev | clean | photo_only | neckline | redacted | 710 | 94.8% (92.8% to 96.6%) | 94.8% | 83.2% | 70.8% | 0.0% |
| m0_dev_photo_only | dev | clean | photo_only | sleeve_length | redacted | 1,063 | 93.4% (91.7% to 95.0%) | 93.4% | 91.0% | 85.5% | 0.0% |
| m0_dev_text_only | dev | clean | text_only | neckline | redacted | 710 | 76.2% (71.8% to 80.5%) | 76.2% | 69.7% | 70.8% | 0.0% |
| m0_dev_text_only | dev | clean | text_only | sleeve_length | redacted | 1,063 | 73.5% (70.3% to 76.5%) | 73.5% | 66.0% | 85.5% | 0.0% |

## Stated confidence

| Run | Split | Subset | Mode | Attribute | Text | Brier | Brier, invalid = 1 | ECE | AUROC | Risk–coverage area | Mean | Distinct values | Most common (share) |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| m0_dev | dev | clean | photo_text | neckline | full | 0.028 | 0.028 | 0.022 | 0.504 | 0.027 | 0.950 | 2 | 0.95 (99.3%) |
| m0_dev | dev | clean | photo_text | neckline | redacted | 0.043 | 0.043 | 0.007 | 0.517 | 0.043 | 0.950 | 3 | 0.95 (99.6%) |
| m0_dev | dev | clean | photo_text | sleeve_length | full | 0.040 | 0.040 | 0.008 | 0.504 | 0.040 | 0.950 | 2 | 0.95 (99.2%) |
| m0_dev | dev | clean | photo_text | sleeve_length | redacted | 0.060 | 0.060 | 0.014 | 0.501 | 0.063 | 0.950 | 2 | 0.95 (99.8%) |
| m0_dev_photo_only | dev | clean | photo_only | neckline | redacted | 0.049 | 0.049 | 0.002 | 0.514 | 0.051 | 0.950 | 2 | 0.95 (99.9%) |
| m0_dev_photo_only | dev | clean | photo_only | sleeve_length | redacted | 0.062 | 0.062 | 0.016 | 0.500 | 0.066 | 0.950 | 1 | 0.95 (100.0%) |
| m0_dev_text_only | dev | clean | text_only | neckline | redacted | 0.207 | 0.207 | 0.162 | 0.536 | 0.224 | 0.924 | 5 | 0.95 (81.1%) |
| m0_dev_text_only | dev | clean | text_only | sleeve_length | redacted | 0.211 | 0.211 | 0.156 | 0.601 | 0.221 | 0.888 | 6 | 0.95 (73.9%) |

## Answer-letter probability (temperature 1)

| Run | Split | Subset | Mode | Attribute | Text | ECE | AUROC | Risk–coverage area | Choice position not found |
|---|---|---|---|---|---|---:|---:|---:|---:|
| m0_dev | dev | clean | photo_text | neckline | full | 0.026 | 0.984 | 0.001 | 0 |
| m0_dev | dev | clean | photo_text | neckline | redacted | 0.041 | 0.960 | 0.003 | 0 |
| m0_dev | dev | clean | photo_text | sleeve_length | full | 0.040 | 0.970 | 0.002 | 0 |
| m0_dev | dev | clean | photo_text | sleeve_length | redacted | 0.063 | 0.938 | 0.006 | 0 |
| m0_dev_photo_only | dev | clean | photo_only | neckline | redacted | 0.046 | 0.969 | 0.003 | 0 |
| m0_dev_photo_only | dev | clean | photo_only | sleeve_length | redacted | 0.065 | 0.918 | 0.008 | 0 |
| m0_dev_text_only | dev | clean | text_only | neckline | redacted | 0.229 | 0.740 | 0.131 | 0 |
| m0_dev_text_only | dev | clean | text_only | sleeve_length | redacted | 0.239 | 0.734 | 0.132 | 0 |

## Accuracy by category

Pooled accuracy is dominated by T-shirts, so each category is shown on its own.

| Run | Split | Subset | Mode | Attribute | Text | Category | n | Accuracy |
|---|---|---|---|---|---|---|---:|---:|
| m0_dev | dev | clean | photo_text | neckline | full | Tshirts | 620 | 99.0% |
| m0_dev | dev | clean | photo_text | neckline | full | Tops | 90 | 84.4% |
| m0_dev | dev | clean | photo_text | neckline | redacted | Tshirts | 620 | 98.9% |
| m0_dev | dev | clean | photo_text | neckline | redacted | Tops | 90 | 72.2% |
| m0_dev | dev | clean | photo_text | sleeve_length | full | Tshirts | 634 | 99.8% |
| m0_dev | dev | clean | photo_text | sleeve_length | full | Shirts | 264 | 85.6% |
| m0_dev | dev | clean | photo_text | sleeve_length | full | Tops | 165 | 97.0% |
| m0_dev | dev | clean | photo_text | sleeve_length | redacted | Tshirts | 634 | 99.4% |
| m0_dev | dev | clean | photo_text | sleeve_length | redacted | Shirts | 264 | 78.0% |
| m0_dev | dev | clean | photo_text | sleeve_length | redacted | Tops | 165 | 96.4% |
| m0_dev_photo_only | dev | clean | photo_only | neckline | redacted | Tshirts | 620 | 98.7% |
| m0_dev_photo_only | dev | clean | photo_only | neckline | redacted | Tops | 90 | 67.8% |
| m0_dev_photo_only | dev | clean | photo_only | sleeve_length | redacted | Tshirts | 634 | 99.5% |
| m0_dev_photo_only | dev | clean | photo_only | sleeve_length | redacted | Shirts | 264 | 76.5% |
| m0_dev_photo_only | dev | clean | photo_only | sleeve_length | redacted | Tops | 165 | 97.0% |
| m0_dev_text_only | dev | clean | text_only | neckline | redacted | Tshirts | 620 | 78.4% |
| m0_dev_text_only | dev | clean | text_only | neckline | redacted | Tops | 90 | 61.1% |
| m0_dev_text_only | dev | clean | text_only | sleeve_length | redacted | Tshirts | 634 | 81.2% |
| m0_dev_text_only | dev | clean | text_only | sleeve_length | redacted | Shirts | 264 | 74.2% |
| m0_dev_text_only | dev | clean | text_only | sleeve_length | redacted | Tops | 165 | 42.4% |

## Evidence sensitivity (full minus redacted text)

Both drops use the items valid in both conditions; the invalid rates are over all items.

| Run | Split | Subset | Mode | Attribute | Items | Valid in both | Confidence drop | Accuracy drop | Invalid, full | Invalid, redacted |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| m0_dev | dev | clean | photo_text | neckline | 710 | 710 | 0.000 | 1.7% | 0.0% | 0.0% |
| m0_dev | dev | clean | photo_text | sleeve_length | 1,063 | 1,063 | 0.000 | 2.3% | 0.0% | 0.0% |

## Invalid outputs by reason

- m0_dev / dev / clean / photo_text / neckline / full: none
- m0_dev / dev / clean / photo_text / neckline / redacted: none
- m0_dev / dev / clean / photo_text / sleeve_length / full: none
- m0_dev / dev / clean / photo_text / sleeve_length / redacted: none
- m0_dev_photo_only / dev / clean / photo_only / neckline / redacted: none
- m0_dev_photo_only / dev / clean / photo_only / sleeve_length / redacted: none
- m0_dev_text_only / dev / clean / text_only / neckline / redacted: none
- m0_dev_text_only / dev / clean / text_only / sleeve_length / redacted: none
