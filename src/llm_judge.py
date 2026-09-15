import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.ollama_client import ollama_client


class LLMJudgeScore(BaseModel):
    helpfulness: int = Field(ge=1, le=5, description="1-5: Problem resolution and accuracy for Apple products")
    tone: int = Field(ge=1, le=5, description="1-5: Warm, professional AppleSupport customer service voice")
    grounding: int = Field(ge=1, le=5, description="1-5: Hallucination check against retrieved historical precedents")
    conciseness: int = Field(ge=1, le=5, description="1-5: Crisp, concise fit for Twitter (<280 chars)")
    rationale: str = Field(description="Summary justifying the sub-scores")


class LLMJudge:
    """
    Independent LLM-as-a-Judge for customer support reply quality.
    Uses config.OLLAMA_JUDGE_MODEL ('qwen3:8b') to eliminate self-preference bias
    against the generation model ('llama3.1:8b').
    """

    RUBRIC_SYSTEM_PROMPT = """You are an impartial, senior quality-assurance evaluator for Apple Customer Support on Twitter (@AppleSupport).
You will be provided with:
1. The Customer's Message.
2. The Historical Resolution Precedents retrieved from proven support archives.
3. The AI Support Agent's Drafted Reply.

Evaluate the AI Agent's Drafted Reply across four distinct dimensions on a strict 1 to 5 integer scale:

### 1. Helpfulness & Correctness (1-5)
- 1: Harmful, misleading, or completely incorrect advice.
- 2: Minimally relevant; fails to address root issue.
- 3: Generic, boilerplate advice (e.g., "restart device") without specific diagnosis.
- 4: Accurate troubleshooting steps or actionable guidance for Apple devices/services.
- 5: Highly specific, precise, actionable resolution directly addressing the issue.

### 2. Tone Match to Brand (@AppleSupport) (1-5)
- 1: Hostile, rude, snarky, or completely inappropriate.
- 2: Cold, excessively robotic, or dismissive.
- 3: Neutral corporate customer service tone.
- 4: Warm, polite, empathetic, classic Apple Support voice ("We'd be happy to take a closer look").
- 5: Exemplary brand voice: reassuring, empathetic, patient, and exceptionally professional.

### 3. Grounding & Hallucination Check (1-5)
CRITICAL: Check whether the reply invents details, policies, fake URLs, or non-existent steps NOT supported by the retrieved historical precedents or standard Apple verified documentation.
- 1: Severe hallucination (fabricated links, non-existent Apple policies, fake refund promises).
- 2: Substantial ungrounded assertions or contradictory advice.
- 3: Plausible generic advice, but claims certain specifics not found in the precedents.
- 4: Well-grounded; sticks closely to proven historical resolution patterns.
- 5: Perfectly grounded; strictly faithful to retrieved precedents without any hallucinated details.

### 4. Conciseness & Format Fit (1-5)
- 1: Excessively verbose, rambling, or exceeds Twitter's 280-character limit.
- 2: Wordy, contains redundant boilerplate.
- 3: Acceptable length, but could be trimmed for faster mobile reading.
- 4: Crisp, well-structured, easy to scan on a mobile device.
- 5: Exceptionally concise, impactful, and direct (<200 characters).

You MUST return ONLY a valid JSON object with integer keys:
{
  "helpfulness": <int 1-5>,
  "tone": <int 1-5>,
  "grounding": <int 1-5>,
  "conciseness": <int 1-5>,
  "rationale": "<concise explanation highlighting grounding and correctness>"
}
Do NOT output markdown code fences, headers, or any text outside the JSON object."""

    def __init__(self, judge_model: str = config.OLLAMA_JUDGE_MODEL):
        self.judge_model = judge_model

    def evaluate(
        self,
        customer_message: str,
        drafted_reply: str,
        retrieved_precedents: Optional[List[Dict[str, Any]]] = None,
        intent_label: Optional[str] = None,
    ) -> LLMJudgeScore:
        """
        Score a drafted reply against the 4 rubric dimensions.
        """
        # Format precedents
        precedent_text = "No historical precedents were retrieved."
        if retrieved_precedents:
            lines = []
            for i, p in enumerate(retrieved_precedents[:3], 1):
                c_msg = p.get("customer_text", "").strip()
                b_reply = p.get("brand_reply", "").strip()
                lines.append(f"Precedent #{i}:\n  Past Customer: \"{c_msg}\"\n  Past Brand Resolution: \"{b_reply}\"")
            precedent_text = "\n\n".join(lines)

        user_prompt = f"""EVALUATION CASE:

Customer Message:
"{customer_message}"

Detected Intent: {intent_label or 'Unknown'}

Retrieved Historical Precedents:
{precedent_text}

AI Agent's Drafted Reply:
"{drafted_reply}"

Evaluate this reply according to the rubric and return the JSON scores:"""

        raw_response = ollama_client.generate(
            prompt=user_prompt,
            system_prompt=self.RUBRIC_SYSTEM_PROMPT,
            model=self.judge_model,
            temperature=0.1,
            max_tokens=300,
        )

        return self._parse_response(raw_response)

    def _parse_response(self, raw: str) -> LLMJudgeScore:
        """Parse and clamp JSON response."""
        # Find JSON object in response
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                def clamp(val, default=3):
                    try:
                        v = int(val)
                        return max(1, min(5, v))
                    except Exception:
                        return default

                return LLMJudgeScore(
                    helpfulness=clamp(data.get("helpfulness", 3)),
                    tone=clamp(data.get("tone", 4)),
                    grounding=clamp(data.get("grounding", 3)),
                    conciseness=clamp(data.get("conciseness", 4)),
                    rationale=str(data.get("rationale", "Evaluated by independent LLM judge.")).strip(),
                )
            except Exception as e:
                pass

        # Fallback if unparseable
        return LLMJudgeScore(
            helpfulness=3,
            tone=3,
            grounding=3,
            conciseness=3,
            rationale=f"Fallback score: unable to parse JSON from judge response ({raw[:80]}...)",
        )


# Global instance
llm_judge = LLMJudge()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    sample_cust = "My battery on iPhone 8 drains in 2 hours after updating to iOS 11. What should I do?"
    sample_precedent = [{
        "customer_text": "Battery life terrible after 11.0.3 update",
        "brand_reply": "We are here to help. Check Settings > Battery to see app usage, and try restarting your device.",
    }]
    sample_reply = "We'd like to help with your battery. Check Settings > Battery to see which apps are using power, and let us know if restarting helps."

    print(f"Running test evaluation with judge model: {llm_judge.judge_model}...")
    score = llm_judge.evaluate(sample_cust, sample_reply, sample_precedent, intent_label="os_update_battery_drain")
    print(f"Helpfulness: {score.helpfulness}/5")
    print(f"Tone:        {score.tone}/5")
    print(f"Grounding:   {score.grounding}/5")
    print(f"Conciseness: {score.conciseness}/5")
    print(f"Rationale:   {score.rationale}")
