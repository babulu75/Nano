from Brain.local import LocalBrain
from Brain.memory import ConversationMemory
from Brain.requirement_classifier import (
    RequirementClassificationError,
    RequirementClassifier,
)
import re
from urllib.parse import urlsplit
from tools.browser.chrome_manager import ChromeManager
from tools.browser.research_pipeline import BrowserResearchPipeline, ResearchError, format_evidence
from config import MEMORY_FILE, MAX_MEMORY_TURNS, debug_log


class NanoAssistant:
    # Feature: Nano request orchestration — Nano v0.4 — Purpose: route each command through local reasoning, reusable tools, evidence validation, and memory.

    def __init__(self):

        self.classifier = RequirementClassifier()
        self.brain = LocalBrain()
        # Feature: shared browser service — Nano v0.4 — Purpose: let browser tools reuse one Chrome session for searches, mail, and future tab-based tasks.
        self.browser = ChromeManager()
        self.researcher = BrowserResearchPipeline(chrome_manager=self.browser)
        self.last_classification = None
        self.memory = ConversationMemory(
            max_turns=MAX_MEMORY_TURNS,
            persist_path=MEMORY_FILE,
        )

    async def process(self, text: str) -> str:

        # Feature: centralized local/web routing — Nano v0.4 — Purpose: use browser research only when the v0.3 classifier requires current or verified facts.
        debug_log(f"Request received; chars={len(text)}.")
        try:
            self.last_classification = self.classifier.classify(text)
        except RequirementClassificationError as error:
            debug_log(f"Requirement classification failed: {error}")
            return f"[Nano] Requirement classification failed: {error}"

        history = self.memory.get_ollama_history()

        tool_context = None
        research_bundle = None
        if self.last_classification["web_required"]:
            debug_log("Starting browser research in the shared Chrome session.")
            try:
                research_bundle = await self.researcher.research(text)
            except ResearchError as error:
                debug_log(f"Browser research failed: {error}")
                return f"[Nano] Web research could not verify this request: {error}"
            tool_context = format_evidence(research_bundle)
            debug_log(f"Browser research succeeded; readable_sources={len(research_bundle.evidence)}; conflicts={len(research_bundle.conflicts)}.")
            # A web answer must be grounded in this turn's retrieved pages, not stale
            # or possibly incorrect prior chat history.
            history = []

        if tool_context is None:
            debug_log("Answer route: local Ollama without browser evidence.")
            response = self.brain.generate(text, history=history)
        else:
            debug_log("Answer route: local Ollama with this turn's browser evidence.")
            response = self.brain.generate(text, history=history, tool_context=tool_context)

        # Keep transport/model errors out of the saved conversation context.
        if response.startswith("[Nano]"):
            debug_log("Local model returned an operational error; skipping memory save.")
            return response

        if research_bundle:
            extracted_urls = {
                url.rstrip(".,;:)")
                for item in research_bundle.evidence
                for url in re.findall(r"https?://[^\s<>\]\[\"']+", item.text)
            }
            extracted_urls = {
                url for url in extracted_urls
                if urlsplit(url).hostname and "." in urlsplit(url).hostname
            }
            if not any(url in response for url in extracted_urls):
                # Keep the answer useful if Ollama paraphrases or formats a citation
                # differently from the source links shown by Google AI Mode.
                source_urls = sorted(extracted_urls)[:5] or [
                    item.final_url for item in research_bundle.evidence
                ]
                response += "\n\nSources shown by Google AI Mode:\n" + "\n".join(
                    f"- {url}" for url in source_urls
                )
                debug_log(f"Added Google AI Mode source links; source_count={len(source_urls)}.")

        self.memory.add_exchange(text, response, model="ollama")
        debug_log("Answer completed and saved to conversation memory.")

        return response

    def clear_memory(self):

        self.memory.clear()

    async def shutdown(self):
        """Release Nano-owned browser resources when the assistant session exits."""
        await self.researcher.close()
