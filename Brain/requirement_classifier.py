import json
from typing import Any, Dict

import requests

from config import CLASSIFIER_MODEL, OLLAMA_URL


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
        try:
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
            return classification
        except requests.exceptions.RequestException as exc:
            raise RequirementClassificationError(
                f"Could not reach classifier model '{CLASSIFIER_MODEL}': {exc}"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise RequirementClassificationError(
                "The classifier returned invalid JSON or missing classification fields."
            ) from exc

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
