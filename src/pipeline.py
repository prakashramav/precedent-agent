import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.data_loader import clean_tweet_text
from src.indexer import resolution_index
from src.classifier import classify_intent, ml_classifier, llm_classifier, ClassificationResult
from src.generator import reply_generator, DraftedReply
from src.escalation import evaluate_escalation_policy, EscalationDecision


class PipelineOutput(BaseModel):
    raw_message: str
    cleaned_message: str
    intent_classification: ClassificationResult
    ml_baseline_classification: Optional[ClassificationResult] = None
    retrieved_precedents: List[Dict[str, Any]]
    drafted_reply: DraftedReply
    escalation_decision: EscalationDecision


def run_pipeline(
    customer_message: str,
    classifier_method: str = "llm",
    top_k: int = config.TOP_K_RETRIEVAL,
    compare_baselines: bool = True,
) -> PipelineOutput:
    """
    Run the end-to-end support agent pipeline:
      1. Clean message (strip mentions/URLs while logging traceability)
      2. Intent classification (LLM or ML baseline)
      3. Retrieve top-k historical resolution precedents
      4. Grounded reply generation with precedent citation
      5. Explicit escalation policy evaluation + JSONL audit logging
    """
    # 1. Clean message
    clean_info = clean_tweet_text(customer_message)
    cleaned_text = clean_info["clean_text"] or customer_message

    # 2. Intent classification
    primary_res = classify_intent(cleaned_text, method=classifier_method)
    
    ml_res = None
    if compare_baselines:
        # Run ML baseline as well for auditable side-by-side comparison
        try:
            ml_res = ml_classifier.predict(cleaned_text)
        except Exception as e:
            print(f"ML baseline prediction note: {e}")

    # 3. Retrieve historical precedents
    if not resolution_index.is_built() and resolution_index.vectors is None:
        resolution_index.build_index()
    elif resolution_index.vectors is None:
        resolution_index.load_index()

    precedents = resolution_index.search(cleaned_text, top_k=top_k)
    top_similarity = precedents[0]["similarity_score"] if precedents else 0.0

    # 4. Generate grounded reply
    drafted = reply_generator.generate_reply(
        customer_message=cleaned_text,
        retrieved_precedents=precedents,
        intent_label=primary_res.intent,
    )

    # 5. Escalation decision
    escalation = evaluate_escalation_policy(
        customer_message=customer_message,
        intent=primary_res.intent,
        risk_tier=primary_res.risk_tier,
        confidence=primary_res.confidence,
        top_similarity=top_similarity,
        insufficient_context=drafted.insufficient_context,
    )

    return PipelineOutput(
        raw_message=customer_message,
        cleaned_message=cleaned_text,
        intent_classification=primary_res,
        ml_baseline_classification=ml_res,
        retrieved_precedents=precedents,
        drafted_reply=drafted,
        escalation_decision=escalation,
    )


if __name__ == "__main__":
    test_msg = "My AirPods won't connect to my iPhone after the iOS update. Any fix?"
    print(f"Processing message: \"{test_msg}\"")
    output = run_pipeline(test_msg)
    
    print("\n=== PIPELINE RESULT ===")
    print(f"Intent: {output.intent_classification.intent} (Confidence: {output.intent_classification.confidence:.2f}, Risk: {output.intent_classification.risk_tier})")
    print(f"Retrieved Precedents: {len(output.retrieved_precedents)} cases found (Top similarity: {output.escalation_decision.top_similarity})")
    print(f"Drafted Reply:\n  {output.drafted_reply.reply_text}")
    print(f"Escalation Action: [{output.escalation_decision.action.upper()}]")
    print(f"Reasons: {output.escalation_decision.reasons}")
