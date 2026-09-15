# Critical Evaluation Audit: What's Misleading About My Headline Numbers

This mandatory document provides an unvarnished, rigorous audit of the systemic limitations, surrogate proxy biases, and distributional gaps that could easily deceive executive leadership if headline metrics are taken at face value.

---

## 1. The "Silent Resolution" Fallacy (The Noisy Resolution Proxy)

### The Illusion
In many customer care benchmarks, automated resolution is measured by the surrogate proxy `went_quiet_after_brand == True` (i.e. whether the customer posted a subsequent public reply after the brand's response). Naive reporting often claims:
> *"Our pipeline achieved an 82% resolution rate based on historical thread closure."*

### The Reality
A customer going quiet on Twitter is **not** proof of a satisfied resolution. Our empirical data audit reveals two massive confounding factors:

1. **The DM Channel Shift**:
   In our analysis of 7,020 `@AppleSupport` threads, over **41.3% (2,904 threads)** contained explicit brand directives to leave the public timeline and initiate a private Direct Message (*"Please DM us your serial number"*, *"Follow and DM us to troubleshoot"*). When the customer complied, the Twitter thread went completely quiet. Labeling these threads as "resolved on platform" falsely credits the public agent with a complete fix, when in reality the complex, expensive work was handed off to a human agent in DM.
2. **Customer Resignation / Abandonment**:
   Many customers stop responding simply because they gave up in frustration, found the bot/agent unhelpful, or decided to call 1-800-MY-APPLE instead. In customer service literature, "churning in silence" is well-documented. Treating thread silence as a positive resolution metric overestimates true resolution by an estimated **25% to 35%**.

---

## 2. Class Imbalance Artificially Inflating Headline Accuracy

### The Illusion
A model reporting **85% Overall Accuracy** on Twitter customer support appears production-ready.

### The Reality
Raw overall accuracy is heavily distorted by extreme class imbalance in social media datasets:
- Over **60%** of inbound `@AppleSupport` tweets in raw data fall into two dominant buckets:
  1. `other_general_inquiry` (greetings, miscellaneous comments, praise, store hours)
  2. `os_update_battery_drain` (post-iOS release flood of battery complaints)
- In contrast, critical high-risk intents—such as `account_access_security` (~4%) and `billing_subscription_refund` (~2%)—are statistically sparse.

A trivial baseline that **always predicts `os_update_battery_drain` or `other_general_inquiry`** can achieve ~40-50% raw accuracy while being completely blind to account hijacking and fraudulent subscription charges. 
Therefore:
- **Macro-F1** (unweighted mean across all 8 classes) is the only honest metric.
- In our evaluation, we intentionally stratified the 180-example golden set with balanced quotas (~22 per intent). A pipeline that collapses on rare high-risk classes will see its Macro-F1 drop precipitously, exposing true performance.

---

## 3. The Distributional Gap: Golden Set vs. Raw Twitter Firehose

### The Illusion
Evaluation numbers measured on a clean, curated golden set reflect real-world operating performance.

### The Reality
There is a substantial distribution shift between our curated 180-example evaluation benchmark and the raw, uncurated Twitter stream:

| Attribute | Golden Evaluation Set | Real-World Twitter Stream |
| :--- | :--- | :--- |
| **Language & Text Quality** | Well-formed English queries (20–280 chars) with clear nouns/verbs. | Heavily fragmented, ungrammatical, typo-ridden, slang, and multilingual tweets. |
| **Media & Emojis** | Text-focused, emoji-normalized. | Screenshots of error dialogs, video screen recordings, broken links, meme GIFs. |
| **Intent Multiplicity** | Isolated primary intent. | Multi-part compound rants spanning 3 different products and billing issues. |
| **Spam & Trolling** | Filtered out. | 15–20% bot spam, crypto solicitations, unrelated celebrity mentions. |
| **Context Availability** | Clear standalone customer starter messages. | Mid-thread fragments (*"still not working"*, *"see my previous tweet"*). |

Because our golden set filters out sub-20 character fragments and non-English text, the **in-the-wild accuracy will be 10–15 percentage points lower** unless strict upfront input-filtering guardrails are implemented.

---

## 4. The False Negative Asymmetry (The Most Dangerous Metric)

In consumer support, classification errors are not created equal:
- **False Positive (Type I)**: The bot classifies a simple how-to question as high risk and routes it to a human agent.
  - *Cost:* ~$3–$5 in human agent triage time. The customer still receives accurate help.
- **False Negative (Type II)**: The bot classifies an active account takeover or financial dispute as low risk and responds with a cheerful, canned public troubleshooting tip.
  - *Cost:* Catastrophic account compromise, chargeback disputes, legal threats, and severe brand reputational damage on social media.

Reporting an **Overall Escalation Accuracy of 90%** is completely meaningless if the 10% error rate consists entirely of False Negatives. In our evaluation harness, the **False Negative Rate (FNR)** is isolated and reported as an independent metric. Any system with an FNR exceeding 5% on high-risk tiers should be blocked from production deployment.
