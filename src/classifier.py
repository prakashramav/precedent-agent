import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import joblib
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.ollama_client import ollama_client
from src.taxonomy import get_taxonomy, DEFAULT_TAXONOMY


class ClassificationResult(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str
    risk_tier: str
    rationale: Optional[str] = None


class LLMClassifierOutput(BaseModel):
    intent: str
    confidence: float
    rationale: str


class BaselineMLClassifier:
    """
    Trivial / simple baseline classifier:
    nomic-embed-text embeddings + scikit-learn Logistic Regression / kNN.
    """

    def __init__(self, model_path: Path = config.CLASSIFIER_MODEL_PATH):
        self.model_path = model_path
        self.classifier: Optional[LogisticRegression] = None
        self.classes: List[str] = []

    def is_trained(self) -> bool:
        return self.model_path.exists()

    def train_or_load(self):
        if self.is_trained():
            self.load()
        else:
            self.train()

    def train(self, samples_per_intent: int = 15):
        """
        Train logistic regression using intent seed phrases and historical customer
        queries matching keywords to bootstrap labeled data.
        """
        taxonomy = get_taxonomy()
        train_texts = []
        train_labels = []

        # Bootstrap seed examples from taxonomy keywords & descriptions
        for intent_key, intent_info in taxonomy.items():
            # Add description
            train_texts.append(intent_info["description"])
            train_labels.append(intent_key)

            # Add keyword combinations
            keywords = intent_info.get("sample_keywords", [])
            for kw in keywords:
                train_texts.append(f"My issue is with {kw}")
                train_labels.append(intent_key)
                train_texts.append(f"Can you help me with {kw}?")
                train_labels.append(intent_key)

        # Also pull from historical threads if available
        if config.PROCESSED_THREADS_PATH.exists():
            with open(config.PROCESSED_THREADS_PATH, "r", encoding="utf-8") as f:
                intent_counts = {k: 0 for k in taxonomy}
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    text = item.get("initial_customer_text", "")
                    if len(text) < 15:
                        continue
                    text_lower = text.lower()

                    for intent_key, intent_info in taxonomy.items():
                        if intent_counts[intent_key] >= samples_per_intent:
                            continue
                        # Match keywords
                        matched = any(kw in text_lower for kw in intent_info.get("sample_keywords", []))
                        if matched:
                            train_texts.append(text)
                            train_labels.append(intent_key)
                            intent_counts[intent_key] += 1
                            break

        print(f"Training Baseline ML Classifier on {len(train_texts)} examples across {len(taxonomy)} intents...")
        embeddings = np.array(ollama_client.get_embeddings_batch(train_texts, batch_size=25))
        
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(embeddings, train_labels)

        self.classifier = clf
        self.classes = list(clf.classes_)

        # Persist model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": clf, "classes": self.classes}, self.model_path)
        print(f"Trained and saved baseline classifier to {self.model_path}")

    def load(self):
        data = joblib.load(self.model_path)
        self.classifier = data["classifier"]
        self.classes = data["classes"]

    def predict(self, text: str) -> ClassificationResult:
        if self.classifier is None:
            self.train_or_load()

        emb = np.array(ollama_client.get_embedding(text)).reshape(1, -1)
        probs = self.classifier.predict_proba(emb)[0]
        best_idx = int(np.argmax(probs))
        predicted_intent = self.classes[best_idx]
        confidence = float(probs[best_idx])

        taxonomy = get_taxonomy()
        risk_tier = taxonomy.get(predicted_intent, {}).get("risk_tier", "medium")

        return ClassificationResult(
            intent=predicted_intent,
            confidence=round(confidence, 4),
            method="embedding_logistic_regression",
            risk_tier=risk_tier,
            rationale=f"ML baseline predicted '{predicted_intent}' with {confidence:.1%} softmax probability.",
        )


class PromptedLLMClassifier:
    """
    Prompted zero/few-shot LLM classifier via Ollama REST API.
    Uses config.OLLAMA_LLM_MODEL ('llama3.1:8b').
    """

    def __init__(self, model_name: str = config.OLLAMA_LLM_MODEL):
        self.model_name = model_name

    def predict(self, text: str) -> ClassificationResult:
        taxonomy = get_taxonomy()
        
        # Build concise intent taxonomy prompt
        taxonomy_bullet_points = []
        valid_intents = list(taxonomy.keys())
        for k, v in taxonomy.items():
            taxonomy_bullet_points.append(
                f"- {k}: {v['description']} (Keywords: {', '.join(v.get('sample_keywords', [])[:5])})"
            )
        taxonomy_text = "\n".join(taxonomy_bullet_points)

        system_prompt = (
            "You are an expert customer support classifier for Twitter messages.\n"
            "Your task is to classify an incoming customer message into EXACTLY ONE of the provided intents.\n"
            "You must return ONLY a valid JSON object with keys:\n"
            "  \"intent\": one of the valid intent keys\n"
            "  \"confidence\": float between 0.0 and 1.0 representing your certainty\n"
            "  \"rationale\": a single sentence explaining your classification\n"
            "DO NOT output markdown code fences, comments, or explanations outside the JSON."
        )

        user_prompt = (
            f"Available Intents:\n{taxonomy_text}\n\n"
            f"Customer Message: \"{text}\"\n\n"
            f"Return the JSON object classifying this message:"
        )

        raw_response = ollama_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=200,
        )

        # Parse JSON output
        intent_label = "other_general_inquiry"
        confidence = 0.5
        rationale = "Default fallback"

        try:
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                parsed_intent = str(parsed.get("intent", "")).strip()
                if parsed_intent in valid_intents:
                    intent_label = parsed_intent
                else:
                    # Find closest match
                    for vi in valid_intents:
                        if vi in parsed_intent or parsed_intent in vi:
                            intent_label = vi
                            break

                confidence = float(parsed.get("confidence", 0.75))
                confidence = max(0.0, min(1.0, confidence))
                rationale = str(parsed.get("rationale", "LLM classified message."))
            else:
                rationale = f"Could not parse JSON from response: {raw_response[:60]}"
        except Exception as e:
            rationale = f"Classification parsing exception: {str(e)}"

        risk_tier = taxonomy.get(intent_label, {}).get("risk_tier", "medium")

        return ClassificationResult(
            intent=intent_label,
            confidence=round(confidence, 4),
            method=f"prompted_llm_{self.model_name}",
            risk_tier=risk_tier,
            rationale=rationale,
        )


# Global instances
ml_classifier = BaselineMLClassifier()
llm_classifier = PromptedLLMClassifier()


def classify_intent(text: str, method: str = "llm") -> ClassificationResult:
    """
    Convenience function to classify intent with either 'ml' or 'llm'.
    """
    if method == "ml":
        return ml_classifier.predict(text)
    return llm_classifier.predict(text)


if __name__ == "__main__":
    print("Testing ML Baseline Classifier...")
    ml_classifier.train_or_load()
    test_msg = "My iPhone screen is cracked and unresponsive, how much to fix it?"
    res_ml = ml_classifier.predict(test_msg)
    print(f"ML Result: {res_ml}\n")

    print("Testing Prompted LLM Classifier...")
    res_llm = llm_classifier.predict(test_msg)
    print(f"LLM Result: {res_llm}\n")
