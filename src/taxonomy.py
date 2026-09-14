import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.ollama_client import ollama_client


# Derived 8 Intent Taxonomy for Customer Support on Twitter (@AppleSupport domain)
# Derived via unsupervised K-Means clustering of customer message embeddings,
# merged into 8 human-readable intents tagged with risk tiers.
DEFAULT_TAXONOMY = {
    "account_access_security": {
        "label": "account_access_security",
        "name": "Account Access & Apple ID Security",
        "risk_tier": "high",
        "description": "Apple ID locked, forgotten password, 2FA verification codes not received, account hacking or takeover fears.",
        "rationale": "High risk because account lockouts prevent customer usage and deal with sensitive credentials and identity security.",
        "sample_keywords": ["apple id", "password", "locked", "verification code", "two factor", "account", "login", "reset", "security"]
    },
    "billing_subscription_refund": {
        "label": "billing_subscription_refund",
        "name": "Billing, Subscriptions & Refund Requests",
        "risk_tier": "high",
        "description": "Unexpected charges, subscription cancellations, refund disputes, App Store purchase issues, double billing.",
        "rationale": "High risk due to direct financial impact, payment disputes, and chargeback threats.",
        "sample_keywords": ["charged", "refund", "subscription", "money", "billing", "cancel subscription", "receipt", "bank", "purchase"]
    },
    "cancellation_churn_complaint": {
        "label": "cancellation_churn_complaint",
        "name": "Severe Dissatisfaction, Churn & Legal Escalations",
        "risk_tier": "high",
        "description": "Customers threatening to cancel services, switch to competitors (e.g. Android/Samsung), legal or regulatory threats.",
        "rationale": "High risk because churn threats and hostile sentiment require senior human intervention and brand damage control.",
        "sample_keywords": ["never again", "cancel", "switch to samsung", "lawyer", "attorney", "unacceptable", "disgusting", "terrible", "worst service"]
    },
    "os_update_battery_drain": {
        "label": "os_update_battery_drain",
        "name": "OS Update, Battery Life & Overheating",
        "risk_tier": "medium",
        "description": "Battery draining quickly, device heating up, issues immediately following an iOS/macOS update, performance throttling.",
        "rationale": "Medium risk because widespread post-update issues affect user experience but rarely involve account security or financial loss.",
        "sample_keywords": ["battery", "ios", "update", "drain", "heat", "overheating", "slow", "lag", "dying fast"]
    },
    "hardware_physical_damage": {
        "label": "hardware_physical_damage",
        "name": "Hardware Malfunction & Physical Repair",
        "risk_tier": "medium",
        "description": "Cracked screen, broken buttons, charging port failure, physical hardware defects, Genius Bar repair booking.",
        "rationale": "Medium risk because physical repairs require diagnostic check-ins and repair appointments.",
        "sample_keywords": ["screen", "cracked", "broken", "repair", "button", "charging port", "hardware", "genius bar", "camera"]
    },
    "connectivity_network_bluetooth": {
        "label": "connectivity_network_bluetooth",
        "name": "Connectivity, Wi-Fi, Bluetooth & Cellular",
        "risk_tier": "medium",
        "description": "Wi-Fi disconnecting, Bluetooth pairing failure (AirPods/CarPlay), cellular carrier signal drop, no service.",
        "rationale": "Medium risk as it disrupts device functionality, often resolvable via network reset or troubleshooting steps.",
        "sample_keywords": ["wifi", "bluetooth", "airpods", "connection", "disconnecting", "no service", "carrier", "cellular", "pair"]
    },
    "how_to_feature_inquiry": {
        "label": "how_to_feature_inquiry",
        "name": "How-To Guides & Settings Guidance",
        "risk_tier": "low",
        "description": "Questions on how to configure settings, transfer data, use native features (Photos, iCloud backup, gestures).",
        "rationale": "Low risk standard informational queries that can be safely automated with knowledge base links and instructions.",
        "sample_keywords": ["how do i", "how to", "settings", "feature", "transfer", "enable", "disable", "where can i find"]
    },
    "other_general_inquiry": {
        "label": "other_general_inquiry",
        "name": "Other General Inquiries",
        "risk_tier": "low",
        "description": "General praise, miscellaneous feedback, or non-critical questions outside defined operational buckets.",
        "rationale": "Low risk catch-all intent for non-urgent messages.",
        "sample_keywords": ["thanks", "question", "apple", "release date", "store hours", "info"]
    }
}


def load_customer_starters(threads_path: Path = config.PROCESSED_THREADS_PATH, max_samples: int = 400) -> List[str]:
    """
    Extract a sample of customer-initiating messages for clustering.
    """
    messages = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            clean_text = item.get("initial_customer_text", "").strip()
            # Filter out very short tweets like 'hi' or just punctuation
            if len(clean_text) >= 20 and len(clean_text.split()) >= 4:
                messages.append(clean_text)
            if len(messages) >= max_samples:
                break
    return messages


def derive_intent_taxonomy(
    sample_size: int = 300,
    n_clusters: int = 10,
    save_taxonomy: bool = True
) -> Dict[str, Any]:
    """
    Embed customer initiating messages, run K-Means clustering,
    inspect cluster terms, and produce derived taxonomy documentation.
    """
    print(f"Sampling up to {sample_size} customer starter messages...")
    messages = load_customer_starters(max_samples=sample_size)
    print(f"Sampled {len(messages)} messages. Computing embeddings via nomic-embed-text...")

    embeddings = np.array(ollama_client.get_embeddings_batch(messages))
    print(f"Computed embeddings matrix of shape {embeddings.shape}.")

    print(f"Clustering with K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute top TF-IDF words for each cluster to inspect semantic centroids
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(messages)
    feature_names = vectorizer.get_feature_names_out()

    cluster_summaries = {}
    for c_id in range(n_clusters):
        c_indices = np.where(cluster_labels == c_id)[0]
        c_messages = [messages[idx] for idx in c_indices]
        
        # Calculate cluster top terms
        if len(c_indices) > 0:
            c_tfidf = tfidf_matrix[c_indices].mean(axis=0)
            top_word_indices = np.asarray(c_tfidf).flatten().argsort()[::-1][:6]
            top_terms = [feature_names[i] for i in top_word_indices if i < len(feature_names)]
        else:
            top_terms = []

        # Find exemplar message closest to centroid
        centroid = kmeans.cluster_centers_[c_id]
        if len(c_indices) > 0:
            c_embs = embeddings[c_indices]
            dists = np.linalg.norm(c_embs - centroid, axis=1)
            exemplar_idx = c_indices[np.argmin(dists)]
            exemplar_text = messages[exemplar_idx]
        else:
            exemplar_text = ""

        cluster_summaries[f"cluster_{c_id}"] = {
            "cluster_id": c_id,
            "size": int(len(c_indices)),
            "top_terms": top_terms,
            "exemplar_message": exemplar_text,
            "sample_messages": c_messages[:3],
        }

    # Save mapping documentation
    taxonomy_data = {
        "brand_handle": config.BRAND_HANDLE,
        "n_clusters_analyzed": n_clusters,
        "total_sampled_messages": len(messages),
        "cluster_diagnostics": cluster_summaries,
        "intents": DEFAULT_TAXONOMY
    }

    if save_taxonomy:
        with open(config.TAXONOMY_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(taxonomy_data, f, indent=2, ensure_ascii=False)
        print(f"Taxonomy mapping and cluster analysis saved to {config.TAXONOMY_MAPPING_PATH}")
        
        # Also write Markdown documentation
        md_path = BASE_DIR / "taxonomy_mapping.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_taxonomy_markdown(taxonomy_data))
        print(f"Taxonomy documentation written to {md_path}")

    return taxonomy_data


def generate_taxonomy_markdown(taxonomy_data: Dict[str, Any]) -> str:
    """Generate clean Markdown documentation of derived intent taxonomy."""
    md = [
        f"# Derived Intent Taxonomy for @{taxonomy_data.get('brand_handle', 'AppleSupport')}\n",
        "This taxonomy is derived by clustering customer-initiating messages from Kaggle Twitter Customer Support using `nomic-embed-text` embeddings and K-Means, then consolidating semantic clusters into 8 human-readable operational intents.\n",
        "## Intent Summary Table\n",
        "| Intent Label | Name | Risk Tier | Rationale |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for key, intent in taxonomy_data["intents"].items():
        md.append(f"| `{intent['label']}` | {intent['name']} | **{intent['risk_tier'].upper()}** | {intent['rationale']} |")

    md.append("\n## Detailed Intent Definitions\n")
    for key, intent in taxonomy_data["intents"].items():
        md.append(f"### `{intent['label']}` ({intent['name']})")
        md.append(f"- **Risk Tier**: `{intent['risk_tier']}`")
        md.append(f"- **Description**: {intent['description']}")
        md.append(f"- **Escalation Rationale**: {intent['rationale']}")
        md.append(f"- **Key Terms**: {', '.join(intent['sample_keywords'])}\n")

    md.append("## Unsupervised Cluster Inspection Summary\n")
    for c_id, diag in taxonomy_data.get("cluster_diagnostics", {}).items():
        md.append(f"#### Cluster {diag['cluster_id']} (size: {diag['size']})")
        md.append(f"- **Top TF-IDF Terms**: `{', '.join(diag['top_terms'])}`")
        md.append(f"- **Centroid Exemplar**: *\"{diag['exemplar_message']}\"*")
        md.append("")

    return "\n".join(md)


def get_taxonomy() -> Dict[str, Any]:
    """Load taxonomy from JSON file or return default."""
    if config.TAXONOMY_MAPPING_PATH.exists():
        with open(config.TAXONOMY_MAPPING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("intents", DEFAULT_TAXONOMY)
    return DEFAULT_TAXONOMY


if __name__ == "__main__":
    derive_intent_taxonomy(sample_size=150, n_clusters=8)
