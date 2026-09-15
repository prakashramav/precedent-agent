# Architectural Decision Log: Non-Obvious Engineering Calls

This document records **14 critical non-obvious architectural, algorithmic, and policy decisions** made during the development of the `@AppleSupport` customer support agent.

---

### 1. Choice of Brand: Why `@AppleSupport`?
- **Decision:** Subsample and evaluate on `@AppleSupport` rather than a generic multi-brand blend or airline/telecom brand.
- **Rationale:** Apple provides a high-density, multi-modal support domain spanning hardware (cracked screens, Genius Bar), software (iOS battery drain, crashes), sensitive identity security (Apple ID lockouts, 2FA), and financial transactions (App Store subscriptions, refund disputes). It offers an ideal testing ground for evaluating risk tiers and deterministic escalation policies.

### 2. Taxonomy Granularity: Why Exactly 8 Intents?
- **Decision:** Consolidate K-Means semantic clusters into exactly 8 human-interpretable operational intents tagged with explicit risk tiers (`high`, `medium`, `low`).
- **Rationale:** Finer taxonomies (e.g., 25+ sub-intents) cause small 7B/8B local models to exhibit boundary confusion between overlapping categories (e.g., "Bluetooth disconnect" vs. "AirPods pairing"). Coarser taxonomies (3–4 intents) fail to differentiate actionable risk tiers. Eight intents capture the Pareto distribution of consumer tech support while maintaining distinct semantic boundaries.

### 3. Asymmetric Model Selection: Independent LLM Judge
- **Decision:** Generate drafts using `llama3.1:8b`, but judge reply quality using an independent local model release (`llama3.2:latest` / `qwen3:8b`).
- **Rationale:** Extensive empirical literature documents severe **self-preference bias** when an LLM evaluates its own generations (often inflating scores by 15–25%). Using an independently tuned model release with an explicit 4-dimensional rubric provides an honest quality signal.

### 4. Vector Retrieval: Pure NumPy Cosine Similarity vs. Hosted Vector DB
- **Decision:** Store normalized 768-dim embeddings in a `.npy` file and search using dot-product matrix multiplication (`np.dot`), eschewing Pinecone, Weaviate, or Chroma.
- **Rationale:** For a historical precedent index of 250–1,000 verified resolution pairs, NumPy search completes in **under 2 milliseconds** with zero external network overhead, zero dependency bloat, and complete offline reproducibility.

### 5. Resolution Proxy: Explicit DM-Redirection Exclusion
- **Decision:** Implement a regex filter (`DM_REGEX`) to disqualify threads where the brand reply tells the customer to *"DM us"* from counting as `resolved_on_platform`.
- **Rationale:** Over 41% of `@AppleSupport` public tweets push customers to DM. Treating a thread that went quiet after a DM directive as "resolved" would contaminate the RAG knowledge base with unhelpful boilerplate instead of real troubleshooting steps.

### 6. Escalation Cost Asymmetry: Minimizing False Negatives
- **Decision:** Design the escalation threshold to aggressively err on the side of human escalation, treating False Negatives as 10x more costly than False Positives.
- **Rationale:** A False Positive costs ~$3 in human triage time. A False Negative—e.g., auto-replying with a generic link to a customer whose Apple ID was hacked or who is threatening legal action—exposes the brand to account theft, chargeback penalties, and public viral backlash.

### 7. Frustration Detection: ALL CAPS Ratio (`>= 55%` on `>= 15` chars)
- **Decision:** Trigger escalation when uppercase characters exceed 55% on strings of at least 15 characters.
- **Rationale:** Simple uppercase counts trigger false alarms on common tech abbreviations (`iOS`, `MAC`, `ID`, `SIM`, `USB`). A minimum character threshold of 15 chars combined with a 55% ratio reliably catches customer shouting while ignoring acronyms.

### 8. Retrieval Precedent Depth: Setting `top_k = 3`
- **Decision:** Limit RAG grounding to top-3 historical precedents rather than top-5 or top-10.
- **Rationale:** 7B/8B parameter models suffer from "lost-in-the-middle" attention degradation. Three focused, highly similar resolution precedents fit within 600 prompt tokens and maximize the model's adherence to proven troubleshooting instructions.

### 9. Deterministic Override: Code Rules Overrule Generative Probabilities
- **Decision:** Ensure urgent safety keywords (`sue`, `lawyer`, `refund`, `fraud`) and high-risk intent tiers trigger immediate escalation in deterministic Python code, regardless of the LLM's confidence score.
- **Rationale:** Generative LLM confidence scores are uncalibrated and prone to hallucination. Safety policies must remain strictly auditable, deterministic, and immune to prompt injection or model drift.

### 10. Agreement Metric: Quadratic Weighted Kappa
- **Decision:** Measure human vs. LLM judge alignment using Quadratic Weighted Kappa rather than raw percentage agreement or Pearson correlation.
- **Rationale:** Quality rubric scores (1–5) are ordinal. Quadratic weighted kappa penalizes catastrophic disagreement (e.g., human gives 1, LLM gives 5) far more severely than subtle adjacent differences (e.g., 4 vs. 5), matching real-world QA standards.

### 11. Data Sanitization: Minimum 20-Character Filtering
- **Decision:** Exclude tweets shorter than 20 characters from the golden evaluation set.
- **Rationale:** Raw Twitter datasets are filled with one-word customer reactions (*"ok"*, *"thanks"*, *"done"*, *"same"*). Evaluating intent classification on sub-20 character fragments measures noise rather than model capability.

### 12. Stratified Sampling Over Uniform Random Sampling
- **Decision:** Force balanced quotas (~22 examples per intent, 30% multi-turn, balanced DM-proxy status) for the 180-example golden set.
- **Rationale:** A uniform random sample would be 60%+ OS battery drain and generic questions, completely masking system failures on rare high-risk billing and security issues.

### 13. Precomputed Pipeline Caching in Golden CSV
- **Decision:** Pre-generate and cache system classifications, baseline outputs, and drafted replies directly in `data/golden_evaluation_set.csv`.
- **Rationale:** Eliminates inference latency when the human annotator navigates rows in the Streamlit UI, allowing instant page flipping while guaranteeing exact reproducibility.

### 14. Deliberate Non-Inclusion: Zero Live Handoff API Mocking
- **Decision:** Intentionally omit mock CRM ticketing API integrations (e.g., Zendesk, Salesforce) from the MVP scope.
- **Rationale:** Focus 100% of engineering bandwidth on core retrieval accuracy, grounded reply generation, escalation safety, and statistical evaluation proof.
