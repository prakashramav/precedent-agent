# AI Customer Support Agent (@AppleSupport) - Evaluation & Proof Suite

A grounded, locally-run AI customer support agent for Twitter/X customer service, evaluated on real-world conversation threads from Kaggle's **Customer Support on Twitter** (`thoughtvector/customer-support-on-twitter`) dataset subsampled for **@AppleSupport**.

This repository contains the complete **Core Pipeline (Task 1)** and the **Evaluation, Baselines, Proof Layer & Deliverables (Task 2)**.

---

## Key Highlights & Proof Deliverables

1. **Stratified Golden Evaluation Set (180 Examples)**: Stratified across 8 taxonomy intents, single-turn vs. multi-turn threads, and public platform resolutions vs. threads pushing to DM (`flag_pushed_to_dm`).
2. **Interactive Streamlit Labeling & Human-Judge UI (`app.py`)**: Mode 2 enables human hand-labeling of true intent, escalation safety, annotator notes, and 1–5 reply quality rubric sub-scores.
3. **Comparative Baselines Suite (`src/baselines.py`)**: Evaluates the main agent side-by-side against a **Trivial Baseline** (majority class predictor + static reply) and a **Simple Baseline** (rule/keyword matcher + 8-intent template bank).
4. **Independent LLM-as-a-Judge (`src/llm_judge.py`)**: Evaluates reply quality across 4 rubric dimensions (Helpfulness, Tone, Grounding, Conciseness) using an independent local model (`llama3.2:latest`) to eliminate self-preference bias against `llama3.1:8b`.
5. **Statistical Agreement Metrics (`src/evaluation.py`)**: Computes Cohen's quadratic weighted kappa ($\kappa_w$) and Mean Absolute Error between LLM judge scores and human ground-truth on the 40 quality examples.
6. **False Negative Audit**: Isolates the critical False Negative Rate (bot auto-handles a message requiring human intervention) as an independent safety metric.
7. **Comprehensive Reports & Audits**:
   - [REPORT.md](REPORT.md) - Executive evaluation report with benchmark tables and next-week roadmap.
   - [docs/sampling_and_labeling_methodology.md](docs/sampling_and_labeling_methodology.md) - Sampling stratification and DM regex rules.
   - [docs/failure_analysis.md](docs/failure_analysis.md) - Top 5 recurring failure patterns with real examples and root causes.
   - [docs/misleading_numbers.md](docs/misleading_numbers.md) - Critical audit of noisy resolution proxies, class imbalance, and distribution gap.
   - [docs/decision_log.md](docs/decision_log.md) - 14 non-obvious engineering decisions and policy trade-offs.

---

## Tech Stack
- **Generation LLM**: Ollama local `llama3.1:8b` via REST API (`http://localhost:11434`).
- **Independent Judge LLM**: Ollama local `llama3.2:latest` (or `qwen3:8b` / `qwen2.5:7b-instruct`).
- **Embeddings**: Ollama `nomic-embed-text` (768-dim) hit via REST API (`/api/embed`).
- **Vector Retrieval**: Pure NumPy cosine similarity matrix search (no external vector database).
- **Classifiers & Baselines**: `scikit-learn` (Logistic Regression, TF-IDF, K-Means).
- **Validation & Interface**: `pydantic` v2, `pandas`, `streamlit`.
- **Audit Logging**: Append-only JSONL (`logs/escalation_audit.jsonl`).

---

## Reproduce Everything in Under 15 Minutes

> [!NOTE]
> **Ollama Model Pulling Time Excluded**:
> Pulling model weights depends on your internet bandwidth and is excluded from the 15-minute runtime budget. Ensure Ollama is running and models are pulled beforehand.

### 1. Pull Required Ollama Models
Run these commands once in your terminal:
```bash
# Embedder (768-dim)
ollama pull nomic-embed-text

# Generation LLM (Drafted replies & prompted classification)
ollama pull llama3.1:8b

# Independent Judge LLM (Reply quality evaluation)
ollama pull llama3.2:latest
```

### 2. Environment Setup (~1 minute)
```bash
# Clone or enter directory
cd "Hiver Assignment"

# Create and activate virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Build Historical Resolution Index & Train Classifier (~2 minutes)
```bash
# 1. Download tweets & reconstruct threads
python src/data_loader.py

# 2. Verify 8-intent taxonomy
python src/taxonomy.py

# 3. Build NumPy resolution vector retrieval index
python src/indexer.py

# 4. Train baseline ML classifier
python src/classifier.py
```

### 4. Generate Stratified Golden Evaluation Set (~15 seconds)
Generates the 180-example stratified evaluation set, precomputing baseline predictions, vectorized retrieval, and system outputs:
```bash
python src/create_golden_set.py
```

### 5. Run Automated Unit Tests (~1 second)
Runs the full 17-test suite covering data loaders, indexers, escalation rules, baselines, and evaluation metrics:
```bash
python -m unittest discover -s tests
```

### 6. Run Evaluation & Benchmark Engine (~5 seconds)
Evaluates intent classification, escalation policy (False Negatives), and LLM judge agreement against human ground truth:
```bash
python src/evaluation.py
```

### 7. Launch Interactive Streamlit UI (Final Step)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Use the sidebar to toggle between:
- **`🚀 Live Pipeline Playground`**: Test custom customer tweets end-to-end with real-time classification, RAG retrieval, and policy audit.
- **`✍️ Golden Set Labeling & Human Judge`**: Hand-label true intent, escalation safety, and hand-score reply quality on the 40-example subset.
- **`📊 Evaluation & Benchmarks Dashboard`**: Live comparative metrics table and agreement statistics.

---

## Escalation Policy Rules
An incoming customer tweet is marked for **`ESCALATE`** (human hand-off) if any of the following rules trigger:
1. **High Risk Intent**: `account_access_security`, `billing_subscription_refund`, or `cancellation_churn_complaint`.
2. **Low Classifier Confidence**: Intent confidence `< 0.65`.
3. **Low Precedent Similarity**: Retrieval top-1 similarity `< 0.55` or `insufficient_context: true`.
4. **Urgent / Legal Keywords**: `cancel`, `refund`, `chargeback`, `lawyer`, `sue`, `court`, `fraud`, `scam`, `police`, `unacceptable`.
5. **Customer Frustration (ALL CAPS)**: Message length `>= 15` characters with an uppercase letter ratio `>= 55%`.

Otherwise, the case is marked as **`AUTO_HANDLE`**. Every decision writes to `logs/escalation_audit.jsonl` with an `audit_id` and timestamp.

---

## Project Directory Structure

```
.
├── REPORT.md                                # Executive evaluation report (max 6 pages)
├── config.py                                # System configurations & threshold parameters
├── requirements.txt                         # Dependencies (requests, pandas, numpy, scikit-learn, pydantic, streamlit)
├── app.py                                   # 3-mode Streamlit web app (Playground, Labeling, Dashboard)
├── cli.py                                   # Headless CLI tester
├── taxonomy_mapping.json                    # Derived 8-intent taxonomy & risk tier definitions
├── taxonomy_mapping.md                      # Human-readable intent documentation
├── data/
│   ├── raw/                                 # Subsampled raw tweets CSV
│   ├── processed/                           # Reconstructed threads, resolution pairs, vector indices (.npy)
│   └── golden_evaluation_set.csv            # 180-example stratified evaluation set
├── docs/
│   ├── sampling_and_labeling_methodology.md # Stratification matrix, DM detection regex, labeling protocol
│   ├── failure_analysis.md                  # Top 5 recurring failure patterns & root causes
│   ├── misleading_numbers.md                # Critical audit of noisy proxies, class imbalance, and distribution gap
│   └── decision_log.md                      # 14 non-obvious engineering decisions & trade-offs
├── logs/
│   ├── escalation_audit.jsonl               # Auditable log of all escalation decisions
│   └── evaluation_results.json              # Exported benchmark metrics & agreement stats
├── src/
│   ├── ollama_client.py                     # Plain requests wrapper for Ollama REST API
│   ├── data_loader.py                       # Ingestion, thread reconstruction, text cleaning, resolution proxy
│   ├── taxonomy.py                          # K-Means clustering & intent taxonomy derivation
│   ├── indexer.py                           # NumPy vector retrieval index over historical resolution pairs
│   ├── classifier.py                        # Dual classifier: ML Logistic Regression & Prompted LLM
│   ├── generator.py                         # Grounded reply generator with precedent citations
│   ├── escalation.py                        # Deterministic escalation policy with keyword & frustration rules
│   ├── baselines.py                         # Trivial baseline & Simple keyword/template baseline
│   ├── llm_judge.py                         # Independent LLM-as-a-judge for reply quality (llama3.2:latest)
│   ├── create_golden_set.py                 # Vectorized 180-example golden set generator
│   ├── evaluation.py                        # Automated metrics, confusion matrix, Cohen's kappa engine
│   └── pipeline.py                          # Unified pipeline orchestrator returning Pydantic schemas
└── tests/
    ├── test_data_loader.py                  # Thread reconstruction & text cleaner unit tests
    ├── test_indexer.py                      # Vector normalization & cosine similarity tests
    ├── test_escalation.py                   # Escalation trigger & safety rule unit tests
    ├── test_baselines.py                    # Trivial and simple baseline unit tests
    ├── test_evaluation.py                   # Evaluation metrics & Cohen's kappa unit tests
    └── test_golden_set.py                   # Golden evaluation set schema & stratification tests
```
