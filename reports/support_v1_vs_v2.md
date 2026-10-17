# Prompt comparison

`support` version 1 compared with version 2.

Suite `support_regression`. Dataset `datasets/support.jsonl`. Model `fake-llm`.

Figures in this report are the scores from the runs being compared. They are not a published benchmark and they do not describe a hosted model unless that model produced the runs.

## Scorer means

| Scorer | v1 | v2 | Delta |
| --- | ---: | ---: | ---: |
| exact_reply | 0.0000 | 1.0000 | +1.0000 |
| policy_regex | 0.0000 | 1.0000 | +1.0000 |
| reference_similarity | 0.0284 | 0.4860 | +0.4576 |
| rubric_judge | 0.0000 | 0.8500 | +0.8500 |

## Case scores

| Case | Scorer | v1 | v2 | Delta |
| --- | --- | ---: | ---: | ---: |
| refund-window | exact_reply | n/a | n/a | n/a |
| refund-window | policy_regex | 0.0000 | 1.0000 | +1.0000 |
| refund-window | reference_similarity | 0.0000 | 0.5423 | +0.5423 |
| refund-window | rubric_judge | 0.0000 | 0.5000 | +0.5000 |
| shipping-delay | exact_reply | n/a | n/a | n/a |
| shipping-delay | policy_regex | 0.0000 | 1.0000 | +1.0000 |
| shipping-delay | reference_similarity | 0.1421 | 0.7071 | +0.5650 |
| shipping-delay | rubric_judge | 0.0000 | 1.0000 | +1.0000 |
| password-reset | exact_reply | n/a | n/a | n/a |
| password-reset | policy_regex | 0.0000 | 1.0000 | +1.0000 |
| password-reset | reference_similarity | 0.0000 | 0.5715 | +0.5715 |
| password-reset | rubric_judge | 0.0000 | 1.0000 | +1.0000 |
| damaged-item | exact_reply | n/a | n/a | n/a |
| damaged-item | policy_regex | 0.0000 | 1.0000 | +1.0000 |
| damaged-item | reference_similarity | 0.0000 | 0.0000 | 0.0000 |
| damaged-item | rubric_judge | 0.0000 | 0.7500 | +0.7500 |
| exact-ack | exact_reply | 0.0000 | 1.0000 | +1.0000 |
| exact-ack | policy_regex | 0.0000 | 1.0000 | +1.0000 |
| exact-ack | reference_similarity | 0.0000 | 0.6093 | +0.6093 |
| exact-ack | rubric_judge | 0.0000 | 1.0000 | +1.0000 |

## Usage

| Version | Task tokens | Judge tokens | Latency ms | Cost USD | Cost known |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 166 | 469 | 0.0 | 0.000000 | yes |
| 2 | 337 | 527 | 0.0 | 0.000000 | yes |

## Outputs

### refund-window

Version 1

    Thanks for reaching out. I will look into that and follow up shortly.

Version 2

    Based on our policy: Laptops may be returned within 30 days of delivery. If this does not cover the question, I will escalate.

### shipping-delay

Version 1

    Thanks for reaching out. I will look into that and follow up shortly.

Version 2

    Based on our policy: If a shipment is more than 5 business days late, refund the shipping fee. If this does not cover the question, I will escalate.

### password-reset

Version 1

    Thanks for reaching out. I will look into that and follow up shortly.

Version 2

    Based on our policy: Account security is outside this policy. Escalate to the identity team. If this does not cover the question, I will escalate.

### damaged-item

Version 1

    Thanks for reaching out. I will look into that and follow up shortly.

Version 2

    Based on our policy: Report damaged items within 7 days of delivery and we will replace them. If this does not cover the question, I will escalate.

### exact-ack

Version 1

    Thanks for reaching out. I will look into that and follow up shortly.

Version 2

    Based on our policy: Reply with the single word received. If this does not cover the question, I will escalate.

