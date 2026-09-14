import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.ollama_client import ollama_client


class DraftedReply(BaseModel):
    reply_text: str
    used_precedents: List[int] = Field(default_factory=list, description="1-indexed ranks of precedents cited/used")
    insufficient_context: bool = Field(
        default=False,
        description="Flagged true if none of the retrieved precedents are actually similar enough to be useful",
    )
    grounding_notes: str = Field(default="", description="Explanation of how precedents were utilized or why inadequate")


class GroundedReplyGenerator:
    """
    Grounded Reply Generator:
    Generates customer support replies conditioned on retrieved historical
    resolutions from the same brand, calling Ollama's REST API with plain requests.
    
    NOTE ON LOCAL 7-8B GROUNDING RELIABILITY:
    Local 7-8B parameter models (such as llama3.1:8b or qwen2.5:7b-instruct) will ground 
    less reliably than larger frontier hosted models (e.g., GPT-4o, Claude 3.5 Sonnet, 
    or Gemini 1.5 Pro). They may occasionally hallucinate standard brand platitudes, 
    partially ignore subtle precedent constraints, or misjudge the relevance of noisy precedents. 
    This is expected behavior at this parameter scale and becomes material for the 
    failure-analysis and evaluation section later, not something to engineer around here.
    """

    def __init__(
        self,
        model_name: str = config.OLLAMA_LLM_MODEL,
        brand_handle: str = config.BRAND_HANDLE,
    ):
        self.model_name = model_name
        self.brand_handle = brand_handle

    def generate_reply(
        self,
        customer_message: str,
        retrieved_precedents: List[Dict[str, Any]],
        intent_label: str,
    ) -> DraftedReply:
        """
        Draft a grounded response using incoming query, detected intent,
        and retrieved historical precedents.
        """
        # Format precedents block
        precedents_formatted = []
        for p in retrieved_precedents:
            rank = p.get("rank", 1)
            score = p.get("similarity_score", 0.0)
            cust = p.get("customer_text", "")
            brand = p.get("brand_reply", "")
            precedents_formatted.append(
                f"[Precedent #{rank}] (Similarity: {score:.3f})\n"
                f"  Customer asked: \"{cust}\"\n"
                f"  Brand resolved: \"{brand}\""
            )

        precedents_str = (
            "\n\n".join(precedents_formatted)
            if precedents_formatted
            else "No historical precedents found."
        )

        system_prompt = (
            f"You are the official Twitter customer support agent for @{self.brand_handle}.\n"
            "Your objective is to draft a helpful, professional, and empathetic Twitter reply to the customer.\n"
            "STRICT GROUNDING RULES:\n"
            "1. Base your response on how the brand historically resolved similar issues in the provided precedents.\n"
            "2. If NONE of the retrieved precedents are actually relevant or helpful to the customer's specific problem, "
            "you MUST flag 'insufficient_context': true.\n"
            "3. Keep the tone concise, friendly, and appropriate for social media customer service.\n"
            "4. Return your output ONLY as a JSON object matching this schema:\n"
            "{\n"
            "  \"reply_text\": \"The drafted reply to the customer\",\n"
            "  \"used_precedents\": [1, 2], // Array of precedent numbers (e.g. [1]) that guided your reply, or [] if none\n"
            "  \"insufficient_context\": false, // Set to true if precedents did not adequately cover this issue\n"
            "  \"grounding_notes\": \"Brief explanation of grounding or why context was insufficient\"\n"
            "}\n"
            "DO NOT include markdown fences, comments, or preamble outside the JSON."
        )

        user_prompt = (
            f"Customer Message: \"{customer_message}\"\n"
            f"Classified Intent: {intent_label}\n\n"
            f"Retrieved Historical Precedents:\n"
            f"{precedents_str}\n\n"
            f"Generate the grounded response JSON:"
        )

        raw_response = ollama_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=350,
        )

        # Parse output
        try:
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return DraftedReply(
                    reply_text=str(parsed.get("reply_text", "")).strip(),
                    used_precedents=parsed.get("used_precedents", []),
                    insufficient_context=bool(parsed.get("insufficient_context", False)),
                    grounding_notes=str(parsed.get("grounding_notes", "")).strip(),
                )
        except Exception as e:
            print(f"Error parsing generator output: {e}, using raw fallback")

        # Fallback if JSON format was corrupted
        return DraftedReply(
            reply_text=raw_response.strip(),
            used_precedents=[],
            insufficient_context=True,
            grounding_notes="Raw response fallback due to JSON parsing failure.",
        )


# Global singleton instance
reply_generator = GroundedReplyGenerator()


if __name__ == "__main__":
    from src.indexer import resolution_index

    test_msg = "I updated to iOS 11 and now my phone dies in 2 hours. Help!"
    test_intent = "os_update_battery_drain"
    
    hits = resolution_index.search(test_msg, top_k=3)
    reply = reply_generator.generate_reply(test_msg, hits, test_intent)
    
    print("--- Grounded Reply Output ---")
    print(f"Drafted Reply: {reply.reply_text}")
    print(f"Used Precedents: {reply.used_precedents}")
    print(f"Insufficient Context: {reply.insufficient_context}")
    print(f"Grounding Notes: {reply.grounding_notes}")
