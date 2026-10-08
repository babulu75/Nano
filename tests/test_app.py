import unittest
from unittest.mock import Mock, mock_open, patch
import json

import requests

from Brain.local import think_local
from Brain.memory import ConversationMemory
from Brain.requirement_classifier import (
    CLASSIFIER_SYSTEM_PROMPT,
    RequirementClassificationError,
    RequirementClassifier,
)
from bridge import NanoBridge


class ConversationMemoryTests(unittest.TestCase):
    def test_persists_and_reloads_messages(self):
        path = "memory-test.json"
        file_data = json.dumps(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        with patch("builtins.open", mock_open(read_data=file_data)):
            with patch("os.path.exists", return_value=True):
                loaded = ConversationMemory(max_turns=2, persist_path=path)

        self.assertEqual(
            loaded.get_ollama_history(),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_trims_oldest_turns(self):
        memory = ConversationMemory(max_turns=1)
        memory.add_user_message("old")
        memory.add_assistant_message("old reply")
        memory.add_user_message("new")
        memory.add_assistant_message("new reply")

        self.assertEqual(
            memory.get_ollama_history(),
            [
                {"role": "user", "content": "new"},
                {"role": "assistant", "content": "new reply"},
            ],
        )

    def test_drops_blank_assistant_turn_and_its_prompt_from_history(self):
        memory = ConversationMemory(max_turns=2)
        memory.add_user_message("question with blank reply")
        memory.add_assistant_message("   ")
        memory.add_user_message("working question")
        memory.add_assistant_message("working answer")

        self.assertEqual(
            memory.get_ollama_history(),
            [
                {"role": "user", "content": "working question"},
                {"role": "assistant", "content": "working answer"},
            ],
        )


class LocalBrainTests(unittest.TestCase):
    @patch("Brain.local.requests.post")
    def test_sends_history_and_prompt_to_ollama(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": "answer"}}
        post.return_value = response

        result = think_local("new question", [{"role": "user", "content": "previous"}])

        self.assertEqual(result, "answer")
        self.assertFalse(post.call_args.kwargs["json"]["think"])
        self.assertEqual(
            post.call_args.kwargs["json"]["messages"],
            [
                {"role": "user", "content": "previous"},
                {"role": "user", "content": "new question"},
            ],
        )
        response.raise_for_status.assert_called_once_with()

    @patch("Brain.local.requests.post", side_effect=requests.exceptions.ConnectTimeout)
    def test_timeout_has_actionable_error(self, _post):
        self.assertIn("did not respond", think_local("hello"))

    @patch("Brain.local.requests.post")
    def test_empty_content_returns_visible_diagnostic(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": "  "}}
        post.return_value = response

        self.assertIn("empty answer", think_local("hello"))


class RequirementClassifierTests(unittest.TestCase):
    @patch("Brain.requirement_classifier.requests.post")
    def test_returns_validated_json_and_only_classifies_input(self, post):
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": '{"task_type":"coding","web_required":false,"complexity":"medium"}'
            }
        }
        post.return_value = response

        result = RequirementClassifier().classify("Write a Python function")

        self.assertEqual(
            result,
            {"task_type": "coding", "web_required": False, "complexity": "medium"},
        )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen2.5:0.5b-instruct")
        self.assertEqual(payload["messages"][1]["content"], "Write a Python function")
        self.assertIn("do not answer", payload["messages"][0]["content"].lower())
        self.assertNotIn("qwen3.5:2b-q4_K_M", payload["model"])
        self.assertEqual(payload["format"]["required"], ["task_type", "web_required", "complexity"])

    @patch("Brain.requirement_classifier.requests.post")
    def test_rejects_invalid_classifier_json(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": '{"answer":"42"}'}}
        post.return_value = response

        with self.assertRaises(RequirementClassificationError):
            RequirementClassifier().classify("What is 6 times 7?")


class BridgeTests(unittest.TestCase):
    def test_process_passes_history_and_saves_exchange(self):
        bridge = NanoBridge()
        bridge.memory = ConversationMemory(max_turns=2)
        bridge.classifier = Mock()
        bridge.classifier.classify.return_value = {
            "task_type": "conversation",
            "web_required": False,
            "complexity": "low",
        }
        bridge.brain = Mock()
        bridge.brain.generate.return_value = "hello back"

        self.assertEqual(bridge.process("hello"), "hello back")
        self.assertEqual(
            bridge.memory.get_ollama_history(),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hello back"},
            ],
        )
        bridge.brain.generate.assert_called_once_with("hello", history=[])
        bridge.classifier.classify.assert_called_once_with("hello")
        self.assertEqual(bridge.last_classification["task_type"], "conversation")

    def test_model_error_is_not_saved_as_a_conversation_turn(self):
        bridge = NanoBridge()
        bridge.memory = ConversationMemory(max_turns=2)
        bridge.classifier = Mock()
        bridge.classifier.classify.return_value = {
            "task_type": "conversation",
            "web_required": False,
            "complexity": "low",
        }
        bridge.brain = Mock()
        bridge.brain.generate.return_value = "[Nano] Ollama returned an empty answer."

        result = bridge.process("hello")

        self.assertIn("empty answer", result)
        self.assertEqual(bridge.memory.count(), 0)

    def test_classifier_failure_stops_before_answer_model(self):
        bridge = NanoBridge()
        bridge.classifier = Mock()
        bridge.classifier.classify.side_effect = RequirementClassificationError("invalid JSON")
        bridge.brain = Mock()

        result = bridge.process("hello")

        self.assertIn("classification failed", result)
        bridge.brain.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
