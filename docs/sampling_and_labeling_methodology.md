# Sampling and Labeling Methodology: Golden Evaluation Set

This document details the sampling strategy, stratification axes, data sanitization rules, and human annotation protocol used to construct the **180-example Golden Evaluation Set** for `@AppleSupport` customer care on Twitter.

---

## 1. Objectives & Framing
Automated customer support evaluation on historical Twitter data suffers from three pervasive vulnerabilities:
1. **Class Imbalance**: Massive overrepresentation of generic OS/battery complaints (`os_update_battery_drain`) and miscellaneous greetings (`other_general_inquiry`) conceals catastrophic failure on rare high-risk intents (e.g. `account_access_security`, `billing_subscription_refund`).
2. **The "Silent Resolution" Fallacy**: In naive pipelines, a thread where the customer did not tweet again is counted as "resolved on platform." In reality, many brand replies direct customers to Direct Messages (DM), or the frustrated user simply abandoned Twitter.
3. **Turn-Depth Bias**: Single-turn interactions ("one-and-done") are vastly simpler than multi-turn threads where initial brand troubleshooting failed and the customer pushed back.

To produce an unvarnished, auditable evaluation, our golden set isolates and stratifies across all three vulnerabilities.

---

## 2. Stratification Matrix (180 Total Examples)

The 180 evaluation candidates are drawn deterministically from `data/processed/applesupport_threads.jsonl` (7,020 reconstructed threads) using a multi-dimensional stratification grid:

### Axis 1: Intent Taxonomy (8 Intents)
We allocate a balanced quota to prevent majority-class masking:
- `cancellation_churn_complaint` (High Risk): **20 examples**
- `billing_subscription_refund` (High Risk): **22 examples**
- `account_access_security` (High Risk): **23 examples**
- `hardware_physical_damage` (Medium Risk): **23 examples**
- `connectivity_network_bluetooth` (Medium Risk): **23 examples**
- `os_update_battery_drain` (Medium Risk): **23 examples**
- `how_to_feature_inquiry` (Low Risk): **23 examples**
- `other_general_inquiry` (Low Risk): **23 examples**

### Axis 2: Thread Depth (Single-Turn vs. Multi-Turn)
- **Single-Turn (`total_turns <= 2`)**: ~70% (126 examples). Represents standard initial customer inquiries.
- **Multi-Turn (`total_turns > 2`)**: ~30% (54 examples). Represents complex interactions where the customer responded to brand inquiries, pushed back on troubleshooting, or expressed lingering dissatisfaction.

### Axis 3: Resolution Proxy & DM-Routing Detection
Every candidate thread is categorized into one of three outcome states:
1. **`resolved_on_platform`**: Thread went quiet after brand reply (`went_quiet_after_brand == True`), and the brand reply **did NOT** redirect the customer off-platform.
2. **`likely_pushed_to_dm`**: Thread went quiet after brand reply, but the brand reply contained DM redirection triggers (see Regex below). **These threads are explicitly flagged and separated** to avoid falsely inflating automated resolution metrics.
3. **`unresolved_followup`**: The customer replied again after the brand's response with a persistent issue, frustration, or negative follow-up (`further_customer_complaint == True`).

#### DM Redirection Regex
```python
DM_REGEX = re.compile(
    r"\b(dm|direct message|message us|reach out in dm|dm us|send a dm|private message)\b|"
    r"twitter\.com/messages|apple\.co/dm",
    re.IGNORECASE
)
```

---

## 3. Reply Quality Ground-Truth Subset (40 Examples)

To evaluate grounded reply generation and calibrate the LLM-as-a-judge system:
- Exactly **40 examples** (5 balanced examples per intent) are flagged with `is_quality_subset = True`.
- These 40 examples undergo dual evaluation:
  1. Hand-scored human ground truth across 4 rubric dimensions (1–5 scale).
  2. Independent LLM-as-judge scoring using `qwen3:8b`.
  3. Agreement calculation via Cohen's quadratic weighted kappa.

---

## 4. Human Labeling Protocol & Rubric

Human annotators use the dedicated **Streamlit Golden Set Labeling UI** (`app.py` Mode 2) to record annotations.

### Field 1: `human_true_intent` (Dropdown)
Select exactly one intent from the 8 official taxonomy categories based on the customer's primary inquiry. If a query expresses multiple intents (e.g. "My battery drains and I want a refund"), choose the higher-risk intent (`billing_subscription_refund`).

### Field 2: `human_should_escalate` (Binary Radio)
Annotator marks whether an automated response is safe or if human handoff is mandatory:
- **`Escalate` (Mandatory Human Intervention)**:
  - Account lockout, credential reset, or security risks.
  - Financial disputes, double billing, or refund demands.
  - Legal, regulatory, or severe churn threats.
  - High emotional distress or complex multi-turn edge cases where generic troubleshooting could aggravate the customer.
- **`Auto-Handle` (Safe for AI Automation)**:
  - Standard how-to guides (e.g. "How do I turn on 2FA in settings?").
  - Common hardware repair routing (e.g. "How to book Genius Bar").
  - Standard connectivity / OS battery troubleshooting with known public steps.

### Field 3: `human_notes` (Freeform Text)
Brief explanation highlighting subtle nuances, ambiguities, or reasons for escalation.

### Field 4: Reply Quality Sub-Scores (1 to 5 scale, for the 40 Quality Examples)
For rows where `is_quality_subset == True`, score the pipeline's drafted reply across 4 axes:

| Score | Helpfulness / Correctness | Tone Match to Brand | Grounding (Hallucination Check) | Conciseness |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Irrelevant, harmful, or incorrect advice. | Hostile, rude, or completely unprofessional. | Completely invents fake URLs, false policies, or hallucinated settings. | Rambling, repetitive, or >280 characters. |
| **2** | Minimally relevant; fails to address root issue. | Overly curt or stiffly robotic. | Cites incorrect information with few real details. | Excessively verbose; bury the lead. |
| **3** | Generic, boilerplate troubleshooting. | Standard corporate tone; lacks empathy. | Plausible steps but includes unverified claims. | Acceptable length, minor unnecessary filler. |
| **4** | Accurate troubleshooting and clear next steps. | Warm, polite, standard Apple support voice. | Closely matches retrieved historical precedents. | Concise, direct, well-structured. |
| **5** | Precise, fully actionable, immediate solution. | Exemplary empathy, warmth, and brand professionalism. | 100% faithful to precedents; zero hallucination. | Punchy, crisp, optimal Twitter fit (<200 chars). |

---

## 5. Audit Trail & Data Integrity
- All annotations are saved directly back to `data/golden_evaluation_set.csv`.
- Precomputed pipeline outputs, baseline predictions, and retrieval rankings are stored alongside human labels to ensure zero latency during labeling and full reproducibility.
