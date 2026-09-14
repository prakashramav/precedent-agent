import sys
import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.pipeline import run_pipeline


SAMPLE_TEST_MESSAGES = [
    "My iPhone 8 battery drains from 100% to 20% in just an hour after updating to iOS 11. Can you help?",
    "I was charged twice for Apple Music subscription this month! I need a refund immediately.",
    "THIS SERVICE IS COMPLETE TRASH I AM CANCELING MY CONTRACT AND SUING YOU IN SMALL CLAIMS COURT!",
    "How do I enable iCloud Photo Library backup on my new iPad?",
    "My AirPods won't pair or show up in Bluetooth settings anymore.",
    "My screen is completely shattered and won't turn on. How much does repair cost at Genius Bar?",
]


def main():
    parser = argparse.ArgumentParser(description="CLI Runner for AI Customer Support Pipeline")
    parser.add_argument("--message", type=str, help="Raw customer message to process through the pipeline")
    parser.add_argument("--method", type=str, default="llm", choices=["llm", "ml"], help="Classifier method")
    parser.add_argument("--top_k", type=int, default=config.TOP_K_RETRIEVAL, help="Top-k precedents to retrieve")
    parser.add_argument("--test_samples", action="store_true", help="Run pipeline sequentially on predefined test scenarios")
    parser.add_argument("--json", action="store_true", help="Output full pipeline result as JSON")
    args = parser.parse_args()

    if args.test_samples:
        print(f"=== Running Pipeline on {len(SAMPLE_TEST_MESSAGES)} Test Scenarios ===\n")
        for i, msg in enumerate(SAMPLE_TEST_MESSAGES, 1):
            print(f"--- Test Case #{i} ---")
            print(f"Customer Message: \"{msg}\"")
            output = run_pipeline(msg, classifier_method=args.method, top_k=args.top_k)
            print(f"Intent: {output.intent_classification.intent} (Confidence: {output.intent_classification.confidence:.2f}, Risk: {output.intent_classification.risk_tier})")
            print(f"Top Precedent Score: {output.escalation_decision.top_similarity:.3f}")
            print(f"Drafted Reply: {output.drafted_reply.reply_text}")
            print(f"Escalation Action: [{output.escalation_decision.action.upper()}]")
            print(f"Reasons: {output.escalation_decision.reasons}")
            print()
        return

    message = args.message
    if not message:
        message = SAMPLE_TEST_MESSAGES[0]
        print(f"No message provided. Using sample message: \"{message}\"\n")

    output = run_pipeline(message, classifier_method=args.method, top_k=args.top_k)

    if args.json:
        print(output.model_dump_json(indent=2))
    else:
        print("=" * 60)
        print("CUSTOMER SUPPORT PIPELINE RESULT")
        print("=" * 60)
        print(f"Customer Message: {output.raw_message}\n")
        print(f"1. INTENT CLASSIFICATION ({output.intent_classification.method}):")
        print(f"   Intent:     {output.intent_classification.intent}")
        print(f"   Confidence: {output.intent_classification.confidence:.2%}")
        print(f"   Risk Tier:  {output.intent_classification.risk_tier.upper()}")
        print(f"   Rationale:  {output.intent_classification.rationale}\n")
        
        if output.ml_baseline_classification:
            print(f"   [Baseline ML Model Comparison]:")
            print(f"   Intent:     {output.ml_baseline_classification.intent} ({output.ml_baseline_classification.confidence:.2%})\n")

        print(f"2. TOP-{len(output.retrieved_precedents)} HISTORICAL PRECEDENTS:")
        for p in output.retrieved_precedents:
            print(f"   [{p['rank']}] Similarity: {p['similarity_score']:.4f} (Thread: {p['thread_id']})")
            print(f"       Customer: {p['customer_text']}")
            print(f"       Reply:    {p['brand_reply']}")
        print()

        print("3. DRAFTED GROUNDED REPLY:")
        print(f"   Reply: {output.drafted_reply.reply_text}")
        print(f"   Used Precedents: {output.drafted_reply.used_precedents}")
        print(f"   Insufficient Context: {output.drafted_reply.insufficient_context}")
        print(f"   Grounding Notes: {output.drafted_reply.grounding_notes}\n")

        print(f"4. ESCALATION DECISION: [{output.escalation_decision.action.upper()}]")
        print("   Triggered Rules / Reasons:")
        for r in output.escalation_decision.reasons:
            print(f"   - {r}")
        print("=" * 60)


if __name__ == "__main__":
    main()
