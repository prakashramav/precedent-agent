# AI Customer Support Agent - Core Pipeline

A grounded, locally-run AI customer support agent for Twitter customer service, trained and evaluated on conversation threads from Kaggle's **Customer Support on Twitter** (`thoughtvector/customer-support-on-twitter`) dataset subsampled for **@AppleSupport**.

The agent performs four core tasks:
1. **Cleans & Reconstructs Threads**: Extracts customer issues, brand agent replies, and identifies noisy resolution proxies.
2. **Intent Classification**: Classifies incoming messages into an 8-intent derived taxonomy using both an **embedding + Logistic Regression baseline** and a **Prompted local LLM (`llama3.1:8b`)**.
3. **Historical Resolution Retrieval (RAG grounding)**: Retrieves top-k proven resolutions using cosine similarity over local embeddings (`nomic-embed-text`).
4. **Grounded Reply Generation**: Drafts Twitter-length replies grounded in historical precedents and flags context insufficiency.
5. **Deterministic Escalation Policy**: Evaluates risk tiers, confidence, retrieval similarity, urgent keywords, and emotional distress markers (ALL CAPS ratio), writing every decision to an auditable JSONL log.

---

## Tech Stack
- **LLM**: Ollama running locally (`llama3.1:8b` or `qwen2.5:7b-instruct`) via REST API (`http://localhost:11434`).
- **Embeddings**: Ollama `nomic-embed-text` (768-dim) hit via REST API (`/api/embed`) using plain `requests` (no LangChain / LlamaIndex).
- **Vector Retrieval**: Pure NumPy cosine similarity index (no hosted vector database).
- **Clustering & Baselines**: `scikit-learn` (K-Means, TF-IDF, Logistic Regression).
- **Data & Validation**: `pandas`, `pydantic`.
- **Interface**: `Streamlit` (`app.py`) and a headless CLI runner (`cli.py`).
- **Logging**: Append-only JSONL (`logs/escalation_audit.jsonl`).

---

## Getting Started in Under 15 Minutes

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.
- [Ollama](https://ollama.com/) installed and running.

Pull the required models in your terminal:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Environment Setup
Clone or enter the project directory:
```bash
cd "Hiver Assignment"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install minimal dependencies
pip install -r requirements.txt
```

### 3. Data Ingestion & Index Building (One-Click)
Run the automated data preparation script. This streams a sample of tweets for `@AppleSupport`, reconstructs conversation threads, builds the taxonomy, and creates the vector retrieval index:

```bash
# 1. Download & reconstruct conversation threads
python src/data_loader.py

# 2. Derive & verify intent taxonomy
python src/taxonomy.py

# 3. Build the local historical resolution vector index
python src/indexer.py

# 4. Train the baseline ML classifier
python src/classifier.py
```
*(All 4 steps take ~2-3 minutes total).*

---

## Running the Pipeline

### Option A: Interactive Streamlit UI
Launch the single-page Streamlit web app:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can select pre-loaded tweets from the dataset or enter custom messages to see the intent classification, retrieved precedents, drafted reply, and escalation policy decision.

### Option B: Headless CLI Runner
Test individual messages or batch test scenarios:

```bash
# Run on built-in test suite (6 realistic scenarios)
python cli.py --test_samples

# Run on a custom customer message
python cli.py --message "My iPhone battery drains in 1 hour after iOS update. Help!"

# Output structured JSON
python cli.py --message "Refund me now or I will sue" --json
```

### Option C: Run Automated Unit Tests
```bash
python -m unittest discover -s tests
```

---

## Project Structure

```
.
├── config.py                      # Single source of truth for models, URLs, thresholds, brand handle
├── requirements.txt               # Dependencies (requests, pandas, numpy, scikit-learn, pydantic, streamlit)
├── taxonomy_mapping.json          # Derived 8-intent taxonomy, risk tiers, and centroid diagnostics
├── taxonomy_mapping.md            # Human-readable intent taxonomy documentation
├── app.py                         # Single-file Streamlit demo UI (plain components)
├── cli.py                         # CLI script for batch testing and evaluation
├── data/
│   ├── raw/                       # Subsampled raw CSV tweets
│   └── processed/                 # Reconstructed threads, resolution pairs, vector indices (.npy)
├── logs/
│   └── escalation_audit.jsonl     # Auditable log of all escalation decisions
├── src/
│   ├── ollama_client.py           # Plain requests wrapper for Ollama REST API (/api/embed & /api/generate)
│   ├── data_loader.py             # Ingestion, thread reconstruction, text cleaning, resolution proxy
│   ├── taxonomy.py                # K-Means clustering & intent taxonomy derivation
│   ├── indexer.py                 # NumPy vector retrieval index over historical resolution pairs
│   ├── classifier.py              # Dual classifier: ML Logistic Regression & Prompted LLM
│   ├── generator.py               # Grounded reply generator with precedent citations
│   ├── escalation.py              # Explicit escalation policy with keyword & frustration rules
│   └── pipeline.py                # Unified pipeline orchestrator returning Pydantic schemas
└── tests/
    ├── test_data_loader.py        # Thread reconstruction & text cleaner unit tests
    ├── test_indexer.py            # Vector normalization & cosine similarity tests
    └── test_escalation.py         # Escalation trigger & safety rule unit tests
```

---

## Escalation Policy Rules
An incoming message is marked for **`ESCALATE`** (human hand-off) if any of the following rules trigger:
1. **High Risk Intent**: Intent belongs to a high-risk tier (`account_access_security`, `billing_subscription_refund`, `cancellation_churn_complaint`).
2. **Low Classifier Confidence**: Primary classifier confidence `< 0.65`.
3. **Low Precedent Similarity**: Retrieval top-1 similarity score `< 0.55`, or LLM generator flags `insufficient_context: true`.
4. **Urgent / Legal Keywords**: Contains terms such as `cancel`, `refund`, `lawyer`, `sue`, `court`, `fraud`, `scam`, `police`, `unacceptable`.
5. **Customer Frustration (ALL CAPS)**: Message has `>= 15` characters and an uppercase letter ratio `>= 55%`.

Otherwise, the case is marked as **`AUTO_HANDLE`**. Every decision is recorded in `logs/escalation_audit.jsonl` with an `audit_id` and timestamp.
