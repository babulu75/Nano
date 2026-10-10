import json
import re
from typing import Any, Dict

import requests

from config import CLASSIFIER_MODEL, OLLAMA_URL, debug_log


CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": [
                "conversation",
                "coding",
                "mathematics",
                "factual",
                "reasoning",
                "other",
            ],
        },
        "web_required": {"type": "boolean"},
        "complexity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": ["task_type", "web_required", "complexity"],
    "additionalProperties": False,
}

CLASSIFIER_SYSTEM_PROMPT = """Classify the user's request only; do not answer it. Return one JSON object.
Choose the main task type: conversation for greetings/social chat, coding for programming,
mathematics for calculations or math problems (including multiplication written in words),
factual for questions asking who/what/when/where, reasoning for analysis or planning,
and other otherwise. Set web_required true when the answer needs current, changing, or
externally verifiable facts. If the request says current, latest, today, now, or recent,
web_required must be true. Complexity is low for simple requests, medium for several
steps or constraints, and high for difficult multi-step work.

Examples:
"Hi! How are you?" -> {"task_type":"conversation","web_required":false,"complexity":"low"}
"Write a Python function that returns the square of a number." -> {"task_type":"coding","web_required":false,"complexity":"medium"}
"What is 17 multiplied by 23?" -> {"task_type":"mathematics","web_required":false,"complexity":"low"}
"Who is the current president of France?" -> {"task_type":"factual","web_required":true,"complexity":"low"}"""


class RequirementClassificationError(RuntimeError):
    """Raised when the local classifier cannot return valid requirement JSON."""


class RequirementClassifier:
    def classify(self, request_text: str) -> Dict[str, Any]:
        # Feature: v0.3 requirement classification with v0.4 local identity routing — Nano v0.4 — Purpose: keep ordinary conversation out of web research.
        if self._is_identity_or_conversation(request_text):
            debug_log("Routing decision: identity/conversation request -> local Ollama; Chrome not needed.")
            return {"task_type": "conversation", "web_required": False, "complexity": "low"}
        if self._is_explicit_web_request(request_text):
            debug_log("Routing decision: explicit internet/search request -> browser tool + local Ollama.")
            return {"task_type": "factual", "web_required": True, "complexity": "low"}
        try:
            debug_log(f"Classifier request started; model={CLASSIFIER_MODEL}, request_chars={len(request_text)}.")
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": CLASSIFIER_MODEL,
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": request_text},
                    ],
                    "format": CLASSIFICATION_SCHEMA,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 96},
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()["message"]["content"]
            classification = json.loads(result)
            self._validate(classification)
            debug_log(f"Classifier result: task_type={classification['task_type']}, web_required={classification['web_required']}, complexity={classification['complexity']}.")
            # Small local classifiers can miss explicit time-sensitive wording.
            # Keep v0.3's classifier, but enforce its hard rule in code as a backstop.
            if self._requires_web_by_wording(request_text):
                classification["web_required"] = True
            debug_log(f"Final routing decision: {'web research + local answer' if classification['web_required'] else 'local Ollama only'}.")
            return classification
        except requests.exceptions.RequestException as exc:
            debug_log(f"Classifier request failed: {type(exc).__name__}: {exc}")
            raise RequirementClassificationError(
                f"Could not reach classifier model '{CLASSIFIER_MODEL}': {exc}"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            debug_log(f"Classifier response validation failed: {type(exc).__name__}: {exc}")
            raise RequirementClassificationError(
                "The classifier returned invalid JSON or missing classification fields."
            ) from exc

    @staticmethod
    def _is_identity_or_conversation(request_text: str) -> bool:
        patterns = (
            r"\bwho\s+are\s+you\b", r"\bwhat(?:'s| is) your name\b",
            r"\bwhat can you do\b", r"\bwho made you\b", r"\btell me about yourself\b",
            r"\bare you an? (?:ai|assistant|bot|language model)\b",
            r"\bhow are you\b", r"^(?:hi|hello|hey|thanks|thank you)[!. ]*$",
        )
        return any(re.search(pattern, request_text.strip(), re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _is_explicit_web_request(request_text: str) -> bool:
        # Feature: explicit web-intent routing — Nano v0.4 — Purpose: prevent phrases such as "find this on the internet" from being mistaken for conversation.
        patterns = (
            r"\b(?:search|look\s+up|find|check)\b.{0,100}\b(?:online|internet|web|net)\b",
            r"\b(?:search|look\s+up)\s+(?:for|online|the\s+web|the\s+internet)\b",
            r"\b(?:online|internet|web)\s+(?:search|results|sources)\b",
        )
        return any(re.search(pattern, request_text, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _validate(classification: Any) -> None:
        if not isinstance(classification, dict):
            raise ValueError("Classification must be a JSON object")
        if set(classification) != {"task_type", "web_required", "complexity"}:
            raise ValueError("Classification fields do not match the schema")
        if classification["task_type"] not in {
            "conversation",
            "coding",
            "mathematics",
            "factual",
            "reasoning",
            "other",
        }:
            raise ValueError("Unknown task type")
        if not isinstance(classification["web_required"], bool):
            raise ValueError("web_required must be boolean")
        if classification["complexity"] not in {"low", "medium", "high"}:
            raise ValueError("Unknown complexity")

    @staticmethod
    def _requires_web_by_wording(request_text: str) -> bool:
        current_cues = (
            r"\btoday\b", r"\btonight\b", r"\bnow\b", r"\bright now\b",
            r"\bcurrent(?:ly)?\b", r"\blatest\b", r"\brecent(?:ly)?\b",
            r"\blive\b", r"\bthis (?:morning|afternoon|evening|week|month|year)\b",
            r"\bas of (?:today|now|\d{4})\b", r"\b(?:happening|going on)\b",
            r"\b(?:ceo|president|prime minister|minister|mayor|governor)\b",
        )
        return any(re.search(pattern, request_text, re.IGNORECASE) for pattern in current_cues)
