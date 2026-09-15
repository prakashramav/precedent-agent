import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Brand Handle Configuration
# Target brand handle from Kaggle Twitter Customer Support (e.g. AppleSupport, AmazonHelp, SpotifyCares)
BRAND_HANDLE = os.getenv("BRAND_HANDLE", "AppleSupport")

# Ollama REST API Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Data File Paths
RAW_DATA_PATH = RAW_DATA_DIR / f"{BRAND_HANDLE.lower()}_subsample.csv"
PROCESSED_THREADS_PATH = PROCESSED_DATA_DIR / f"{BRAND_HANDLE.lower()}_threads.jsonl"
RESOLUTION_PAIRS_PATH = PROCESSED_DATA_DIR / f"{BRAND_HANDLE.lower()}_resolution_pairs.jsonl"
INDEX_VECTORS_PATH = PROCESSED_DATA_DIR / f"{BRAND_HANDLE.lower()}_index_vectors.npy"
INDEX_METADATA_PATH = PROCESSED_DATA_DIR / f"{BRAND_HANDLE.lower()}_index_metadata.jsonl"
CLASSIFIER_MODEL_PATH = PROCESSED_DATA_DIR / f"{BRAND_HANDLE.lower()}_classifier.joblib"
TAXONOMY_MAPPING_PATH = BASE_DIR / "taxonomy_mapping.json"
AUDIT_LOG_PATH = LOGS_DIR / "escalation_audit.jsonl"

# Policy & Threshold Configurations
SIMILARITY_THRESHOLD = 0.55  # If top-1 precedent similarity < threshold, flag for escalation
CONFIDENCE_THRESHOLD = 0.65  # If intent classifier confidence < threshold, flag for escalation
HIGH_RISK_TIERS = {"high"}

# Rule-based Urgency & Safety Keywords
URGENT_KEYWORDS = [
    "cancel", "cancellation", "cancelling", "refund", "chargeback", "lawyer", 
    "attorney", "sue", "suing", "lawsuit", "court", "legal", "never again", 
    "fraud", "scam", "stolen", "police", "unacceptable", "terrible", "worst", 
    "disgusting", "pathetic", "furious", "danger", "hazard", "threat"
]

ALL_CAPS_RATIO_THRESHOLD = 0.55
ALL_CAPS_MIN_LENGTH = 15
TOP_K_RETRIEVAL = 3
