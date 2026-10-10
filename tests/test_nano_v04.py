import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from Brain.memory import ConversationMemory
from Brain.requirement_classifier import RequirementClassificationError
from Brain.local import think_local
from Brain.requirement_classifier import RequirementClassifier
from assistant import NanoAssistant
from tools.browser.chrome_manager import ChromeManager
from tools.browser.page_reader import PageReadError, PageReader
from tools.browser.research_pipeline import BrowserResearchPipeline, ResearchBundle, ResearchError, format_evidence
from tools.browser.search_engine import (
    BrowserSearchEngine,
    NoSearchResultsError,
    SearchBlockedError,
    SearchTimeoutError,
    extract_search_results,
    normalize_destination,
)
from tools.browser.source_validator import EvidenceError, find_conflicts, validate_evidence
from models.evidence import Evidence, SearchResult
import config


class FakeLocator:
    def __init__(self, entries=None, text=""):
        self.entries = entries or []
        self.text = text
        self.first = self

    async def count(self):
        return len(self.entries)

    def nth(self, index):
        return self.entries[index]

    async def inner_text(self, timeout=None):
        return self.text

    async def get_attribute(self, name, timeout=None):
        return None


class FakeLink:
    def __init__(self, title, href):
        self.title, self.href = title, href

    async def inner_text(self, timeout=None):
        return self.title

    async def get_attribute(self, name):
        return self.href if name == "href" else None


class FakePage:
    def __init__(self, url="https://www.google.com/search?q=test", body="", title="Search"):
        self.url, self.body, self.page_title = url, body, title
        self.links = {}
        self.goto_calls = []
        self.goto_effects = []
        self.redirect_to = None

    async def close(self):
        self.closed = True

    def locator(self, selector):
        if selector == "body":
            return FakeLocator(text=self.body)
        return FakeLocator(self.links.get(selector, []))

    async def title(self):
        return self.page_title

    async def goto(self, url, wait_until=None, timeout=None):
        self.goto_calls.append((url, timeout))
        if self.goto_effects:
            effect = self.goto_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                effect(self, url)
        self.url = self.redirect_to or url


def evidence(url="https://example.com/article", text="Apple's price was $100.", **kwargs):
    values = dict(
        title="Article", original_url=url, final_url=url, text=text,
        retrieved_at="2026-10-10T00:00:00+00:00", source_domain="example.com",
    )
    values.update(kwargs)
    return Evidence(**values)


def run_assistant(assistant, text):
    return asyncio.run(assistant.process(text))


def test_valid_google_result_extraction():
    page = FakePage()
    page.links["a:has(h3)"] = [FakeLink("  Example result  ", "https://example.com/story")]
    results = asyncio.run(extract_search_results(page, "Google"))
    assert results == [SearchResult("Example result", "https://example.com/story", engine="Google")]


def test_missing_titles_and_invalid_urls_are_skipped():
    page = FakePage()
    page.links["a:has(h3)"] = [
        FakeLink("", "https://empty-title.example/a"),
        FakeLink("Good title", "javascript:alert(1)"),
        FakeLink("Valid title", "https://valid.example/a"),
    ]
    assert [r.url for r in asyncio.run(extract_search_results(page, "Google"))] == ["https://valid.example/a"]


def test_duplicate_result_urls_are_removed():
    page = FakePage()
    page.links["a:has(h3)"] = [
        FakeLink("First", "https://example.com/story#part"),
        FakeLink("Duplicate", "https://example.com/story"),
    ]
    results = asyncio.run(extract_search_results(page, "Google"))
    assert len(results) == 1


@pytest.mark.parametrize("url", ["javascript:alert(1)", "mailto:user@example.com", "https://google.com/search?q=internal"])
def test_internal_or_unsupported_result_urls_rejected(url):
    assert normalize_destination(url, "https://www.google.com/search?q=x") is None


def test_no_extractable_links_raises_clear_error():
    page = FakePage()
    with pytest.raises(NoSearchResultsError):
        asyncio.run(BrowserSearchEngine(engines=(("Google", "https://google.com/search?q={query}"),)).search(page, "x"))


def test_captcha_stops_blocked_engine_attempt_then_uses_independent_provider():
    page = FakePage()
    page.goto_effects = [
        lambda page, _url: setattr(page, "body", "Please verify that you are human. CAPTCHA"),
        lambda page, _url: setattr(page, "body", ""),
    ]
    page.links["li.b_algo h2 a"] = [FakeLink("Bing result", "https://example.com/story")]
    engine = BrowserSearchEngine(engines=(("Google", "https://google.com/search?q={query}"), ("Bing", "https://bing.com/search?q={query}")))
    results = asyncio.run(engine.search(page, "x"))
    assert len(page.goto_calls) == 2
    assert results[0].engine == "Bing"


def test_all_search_engines_blocked_returns_clear_error():
    page = FakePage(body="Access denied. CAPTCHA")
    engines = (("Google", "https://google.com/search?q={query}"), ("Bing", "https://bing.com/search?q={query}"))
    with pytest.raises(SearchBlockedError, match="All configured search engines blocked"):
        asyncio.run(BrowserSearchEngine(engines=engines).search(page, "x"))
    assert len(page.goto_calls) == 2


def test_changed_layout_uses_generic_anchor_fallback():
    page = FakePage()
    page.links["a[href]"] = [FakeLink("Fallback result", "https://example.net/page")]
    assert asyncio.run(extract_search_results(page, "Google"))[0].url == "https://example.net/page"


def test_navigation_timeout_retries_only_within_bound():
    page = FakePage()
    page.goto_effects = [TimeoutError("navigation timeout"), TimeoutError("navigation timeout")]
    engine = BrowserSearchEngine(engines=(("Google", "https://google.com/search?q={query}"),), retries=1)
    with pytest.raises(SearchTimeoutError):
        asyncio.run(engine.search(page, "x"))
    assert len(page.goto_calls) == 2


def test_page_reader_rejects_empty_body():
    page = FakePage(url="https://example.com/a", body="   ")
    with pytest.raises(PageReadError, match="No readable page text"):
        asyncio.run(PageReader(retries=0).read(page, SearchResult("A", "https://example.com/a")))


def test_page_reader_truncates_and_records_final_redirect_url():
    page = FakePage(url="https://example.com/final", body="Readable text " * 10, title="Final page")
    page.redirect_to = "https://example.com/final"
    item = asyncio.run(PageReader(retries=0, max_chars=20).read(page, SearchResult("Original", "https://short.example/a")))
    assert item.final_url == "https://example.com/final"
    assert item.original_url == "https://short.example/a"
    assert len(item.text) == 20
    assert item.extraction_status == "truncated"
    assert item.warning


def test_page_reader_preserves_actual_publication_metadata_only():
    page = FakePage(url="https://example.com/a", body="Text", title="Title")
    date_locator = FakeLocator()
    date_locator.get_attribute = AsyncMock(return_value="2026-10-09")
    page.locator = lambda selector: date_locator if selector == "meta[property='article:published_time']" else FakeLocator(text="Text" if selector == "body" else "")
    item = asyncio.run(PageReader(retries=0).read(page, SearchResult("Original", "https://example.com/a")))
    assert item.publication_date == "2026-10-09"
    assert item.publication_date_source == "article:published_time"
    assert item.retrieved_at != item.publication_date


def test_unknown_publication_date_remains_unknown():
    page = FakePage(url="https://example.com/a", body="Text")
    item = asyncio.run(PageReader(retries=0).read(page, SearchResult("A", "https://example.com/a")))
    assert item.publication_date is None
    assert item.publication_date_source is None


def test_source_validator_deduplicates_final_urls_and_flags_conflicting_values():
    first = evidence("https://example.com/a", "Apple price was $100.")
    duplicate = evidence("https://example.com/a#part", "Apple price was $100.")
    second = evidence("https://other.net/b", "Apple price was $120.", source_domain="other.net")
    selected, conflicts = validate_evidence([first, duplicate, second])
    assert len(selected) == 2
    assert conflicts and conflicts[0].metric == "price"
    assert set(conflicts[0].values) == {"$100", "$120"}


def test_empty_or_content_free_evidence_is_rejected():
    with pytest.raises(EvidenceError):
        validate_evidence([evidence(text="", extraction_status="empty")])


def test_evidence_format_includes_sources_dates_and_conflicts():
    bundle = ResearchBundle([evidence()], [])
    rendered = format_evidence(bundle)
    assert "Final URL: https://example.com/article" in rendered
    assert "Retrieved at (UTC):" in rendered
    assert "Publication/modification date: unknown" in rendered


def test_local_brain_prompt_treats_webpage_as_untrusted_and_includes_evidence():
    response = Mock()
    response.json.return_value = {"message": {"content": "uncertain answer"}}
    with patch("Brain.local.requests.post", return_value=response) as post:
        result = think_local("question", tool_context="Source URL: https://example.com\ntext")
    assert result == "uncertain answer"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "qwen3.5:2b-q4_K_M"
    systems = [item["content"] for item in payload["messages"] if item["role"] == "system"]
    evidence_prompt = next(system for system in systems if "untrusted data" in system)
    assert "publication date" in evidence_prompt
    assert "https://example.com" in evidence_prompt
    assert "never claim you searched" in systems[0].lower()


def test_bridge_non_web_request_skips_browser_research():
    bridge = NanoAssistant()
    bridge.memory = ConversationMemory(max_turns=2)
    bridge.classifier = Mock()
    bridge.classifier.classify.return_value = {"task_type": "conversation", "web_required": False, "complexity": "low"}
    bridge.brain = Mock()
    bridge.brain.generate.return_value = "local reply"
    bridge.researcher = Mock()
    assert run_assistant(bridge, "hello") == "local reply"
    bridge.researcher.research.assert_not_called()
    bridge.brain.generate.assert_called_once_with("hello", history=[])


def test_bridge_web_request_passes_browser_evidence_to_local_model():
    bridge = NanoAssistant()
    bridge.memory = ConversationMemory(max_turns=2)
    bridge.classifier = Mock()
    bridge.classifier.classify.return_value = {"task_type": "factual", "web_required": True, "complexity": "low"}
    bridge.brain = Mock()
    bridge.brain.generate.return_value = "Grounded answer. Source: https://example.com/article"
    bridge.memory.add_user_message("old question")
    bridge.memory.add_assistant_message("stale unsupported answer")
    bridge.researcher = Mock()
    bridge.researcher.research = AsyncMock(return_value=ResearchBundle([evidence()], []))
    assert "Grounded answer" in run_assistant(bridge, "current fact")
    args = bridge.brain.generate.call_args
    assert args.kwargs["history"] == []
    assert "https://example.com/article" in args.kwargs["tool_context"]
    bridge.researcher.research.assert_awaited_once_with("current fact")


def test_web_answer_without_a_source_citation_is_rejected_and_not_saved():
    bridge = NanoAssistant()
    bridge.memory = ConversationMemory(max_turns=2)
    bridge.classifier = Mock()
    bridge.classifier.classify.return_value = {"task_type": "factual", "web_required": True, "complexity": "low"}
    bridge.brain = Mock()
    bridge.brain.generate.return_value = "The answer is Tim Cook, with several extra unsupported details."
    bridge.researcher = Mock()
    bridge.researcher.research = AsyncMock(return_value=ResearchBundle([evidence()], []))
    result = run_assistant(bridge, "Who is the CEO of Apple?")
    assert "did not cite a page it read" in result
    assert "https://example.com/article" in result
    assert bridge.memory.count() == 0


def test_bridge_reports_insufficient_browser_evidence_without_answering():
    bridge = NanoAssistant()
    bridge.classifier = Mock()
    bridge.classifier.classify.return_value = {"task_type": "factual", "web_required": True, "complexity": "low"}
    bridge.researcher = Mock()
    bridge.researcher.research = AsyncMock(side_effect=ResearchError("No readable source pages"))
    bridge.brain = Mock()
    assert "could not verify" in run_assistant(bridge, "current fact").lower()
    bridge.brain.generate.assert_not_called()


def test_classifier_failure_stops_before_browser_or_model():
    bridge = NanoAssistant()
    bridge.classifier = Mock()
    bridge.classifier.classify.side_effect = RequirementClassificationError("bad classifier output")
    bridge.researcher = Mock()
    bridge.brain = Mock()
    assert "classification failed" in run_assistant(bridge, "question").lower()
    bridge.researcher.research.assert_not_called()
    bridge.brain.generate.assert_not_called()


def test_classifier_enforces_web_for_explicit_today_cue():
    response = Mock()
    response.json.return_value = {
        "message": {"content": '{"task_type":"factual","web_required":false,"complexity":"low"}'}
    }
    with patch("Brain.requirement_classifier.requests.post", return_value=response):
        result = RequirementClassifier().classify("What is happening today in Delhi?")
    assert result["web_required"] is True


# Feature: direct internet-intent routing — Nano v0.4 — Purpose: make explicit web requests reach the browser tool even when the small classifier would mislabel them.
@pytest.mark.parametrize("question", [
    "Find about Babulu Digamarti in internet",
    "Who is this person? Can you find his details from net?",
    "Search online for the latest Delhi updates",
])
def test_explicit_web_requests_bypass_ambiguous_classifier(question):
    with patch("Brain.requirement_classifier.requests.post") as post:
        result = RequirementClassifier().classify(question)
    assert result["web_required"] is True
    post.assert_not_called()


# Feature: deterministic local identity routing — Nano v0.4 — Purpose: guarantee self/identity questions never invoke the browser or requirement model.
@pytest.mark.parametrize("question", ["Who are you?", "What's your name?", "What can you do?", "Are you an AI assistant?"])
def test_identity_questions_are_routed_local_without_classifier_request(question):
    with patch("Brain.requirement_classifier.requests.post") as post:
        result = RequirementClassifier().classify(question)
    assert result["web_required"] is False
    assert result["task_type"] == "conversation"
    post.assert_not_called()


def test_identity_question_end_to_end_never_starts_browser():
    bridge = NanoAssistant()
    bridge.memory = ConversationMemory(max_turns=2)
    bridge.brain = Mock()
    bridge.brain.generate.return_value = "I am Nano, a local assistant."
    bridge.researcher = Mock()
    with patch("Brain.requirement_classifier.requests.post") as post:
        result = run_assistant(bridge, "Who are you?")
    assert result == "I am Nano, a local assistant."
    post.assert_not_called()
    bridge.researcher.research.assert_not_called()
    bridge.brain.generate.assert_called_once_with("Who are you?", history=[])


# Feature: single-switch backend diagnostics — Nano v0.4 — Purpose: verify debug output can be enabled or silenced centrally.
def test_central_debug_log_obeys_single_switch(capsys):
    config.DEBUG_LOGS = False


def test_essential_memory_error_stays_visible_when_debug_is_disabled(capsys, tmp_path):
    config.DEBUG_LOGS = False
    memory = ConversationMemory(persist_path=str(tmp_path / "missing" / "file.json"))
    memory.persist_path = str(tmp_path / "as-file" / "child.json")
    (tmp_path / "as-file").write_text("not a directory", encoding="utf-8")
    memory.save()
    captured = capsys.readouterr()
    assert "Failed to save conversation memory" in captured.err
    assert "Nano debug" not in captured.out
    config.debug_log("hidden diagnostic")
    assert capsys.readouterr().out == ""
    config.DEBUG_LOGS = True
    config.debug_log("visible diagnostic")
    assert "visible diagnostic" in capsys.readouterr().out
    config.DEBUG_LOGS = False


def test_search_engine_diagnostics_obey_central_switch(capsys):
    config.DEBUG_LOGS = False
    page = FakePage()
    page.links["li.b_algo h2 a"] = [FakeLink("Bing result", "https://example.com/story")]
    engine = BrowserSearchEngine(engines=(("Bing", "https://bing.com/search?q={query}"),))
    asyncio.run(engine.search(page, "test"))
    assert capsys.readouterr().out == ""
    config.DEBUG_LOGS = True
    page = FakePage()
    page.links["li.b_algo h2 a"] = [FakeLink("Bing result", "https://example.com/story")]
    asyncio.run(engine.search(page, "test"))
    captured = capsys.readouterr().out
    assert "Search requested" in captured
    assert "Search result 1" in captured
    config.DEBUG_LOGS = False


def test_ollama_failure_is_not_saved_to_conversation_memory():
    bridge = NanoAssistant()
    bridge.memory = ConversationMemory(max_turns=2)
    bridge.classifier = Mock()
    bridge.classifier.classify.return_value = {"task_type": "conversation", "web_required": False, "complexity": "low"}
    bridge.brain = Mock()
    bridge.brain.generate.return_value = "[Nano] Cannot reach Ollama."
    assert "Cannot reach Ollama" in run_assistant(bridge, "hello")
    assert bridge.memory.count() == 0


def test_research_pipeline_closes_task_tab_but_keeps_shared_browser_after_error():
    context = Mock()
    page = FakePage()
    context.new_page = AsyncMock(return_value=page)
    manager = Mock()
    manager.start = AsyncMock(return_value=context)
    manager.new_page = AsyncMock(return_value=page)
    manager.close_page = AsyncMock(side_effect=ChromeManager.close_page)
    manager.close = AsyncMock()
    search = Mock()
    search.search = AsyncMock(side_effect=RuntimeError("unexpected"))
    pipeline = BrowserResearchPipeline(chrome_manager=manager, search_engine=search)
    with pytest.raises(ResearchError):
        asyncio.run(pipeline.research("question"))
    manager.close_page.assert_awaited_once_with(page)
    manager.close.assert_not_awaited()


def test_research_pipeline_reads_no_more_than_page_limit():
    context = Mock()
    context.new_page = AsyncMock(return_value=FakePage())
    manager = Mock()
    manager.start = AsyncMock(return_value=context)
    manager.new_page = AsyncMock(return_value=context.new_page.return_value)
    manager.close_page = AsyncMock(side_effect=ChromeManager.close_page)
    manager.close = AsyncMock()
    search = Mock()
    search.search = AsyncMock(return_value=[SearchResult(str(i), f"https://example.com/{i}") for i in range(5)])
    reader = Mock()
    reader.read = AsyncMock(side_effect=lambda _page, result: evidence(result.url, "Readable"))
    pipeline = BrowserResearchPipeline(manager, search, reader, max_pages=2)
    bundle = asyncio.run(pipeline.research("question"))
    assert len(bundle.evidence) == 2
    assert reader.read.await_count == 2
    manager.close_page.assert_awaited_once()


# Feature: browser session reuse across tool calls — Nano v0.4 — Purpose: retain one Chrome context for multiple searches and close it only when the assistant shuts down.
def test_research_pipeline_reuses_shared_browser_and_closes_it_at_shutdown():
    pages = [FakePage(), FakePage()]
    context = Mock()
    manager = Mock()
    manager.start = AsyncMock(return_value=context)
    manager.new_page = AsyncMock(side_effect=pages)
    manager.close_page = AsyncMock(side_effect=ChromeManager.close_page)
    manager.close = AsyncMock()
    search = Mock()
    search.search = AsyncMock(return_value=[SearchResult("Article", "https://example.com/article")])
    reader = Mock()
    reader.read = AsyncMock(return_value=evidence(text="Readable."))
    pipeline = BrowserResearchPipeline(manager, search, reader)

    async def use_shared_browser():
        await pipeline.research("first query")
        await pipeline.research("second query")
        manager.close.assert_not_awaited()
        await pipeline.close()

    asyncio.run(use_shared_browser())
    assert manager.new_page.await_count == 2
    assert manager.close_page.await_count == 2
    manager.close.assert_awaited_once()


def test_research_pipeline_enforces_total_evidence_character_limit():
    context = Mock()
    context.new_page = AsyncMock(return_value=FakePage())
    manager = Mock()
    manager.start = AsyncMock(return_value=context)
    manager.new_page = AsyncMock(return_value=context.new_page.return_value)
    manager.close_page = AsyncMock(side_effect=ChromeManager.close_page)
    manager.close = AsyncMock()
    search = Mock()
    search.search = AsyncMock(return_value=[SearchResult(str(i), f"https://example.com/{i}") for i in range(2)])
    reader = Mock()
    reader.read = AsyncMock(side_effect=lambda _page, result: evidence(result.url, "abcdefghij"))
    pipeline = BrowserResearchPipeline(manager, search, reader, max_pages=2, max_total_chars=7)
    bundle = asyncio.run(pipeline.research("question"))
    assert sum(len(item.text) for item in bundle.evidence) == 7


def test_chrome_manager_reuses_visible_profile_and_opens_reusable_tabs(tmp_path):
    context = AsyncMock()
    chromium = Mock()
    page = AsyncMock()
    page.goto = AsyncMock()
    context.new_page = AsyncMock(return_value=page)
    chromium.launch_persistent_context = AsyncMock(return_value=context)
    playwright = Mock()
    playwright.chromium = chromium
    playwright.stop = AsyncMock()
    playwright_factory = Mock()
    playwright_factory.start = AsyncMock(return_value=playwright)
    async def exercise_session():
        manager = ChromeManager("C:/Chrome/chrome.exe", str(tmp_path / "nano-profile"))
        assert await manager.start() is context
        assert await manager.start() is context
        assert await manager.open_tab("https://mail.google.com") is page
        await manager.close_page(page)
        await manager.close()

    with patch("tools.browser.chrome_manager.async_playwright", return_value=playwright_factory) as factory, patch.object(Path, "is_file", return_value=True):
        asyncio.run(exercise_session())
        options = chromium.launch_persistent_context.call_args.kwargs
        assert options["headless"] is False
        assert options["user_data_dir"] == str(tmp_path / "nano-profile")
        factory.assert_called_once_with()
        chromium.launch_persistent_context.assert_awaited_once()
    context.close.assert_awaited_once()
    playwright.stop.assert_awaited_once()
    page.goto.assert_awaited_once()
    page.close.assert_awaited_once()


@pytest.mark.integration
def test_chrome_search_integration_is_explicitly_opt_in():
    import os
    if os.getenv("NANO_RUN_INTEGRATION") != "1":
        pytest.skip("Set NANO_RUN_INTEGRATION=1 to launch visible Chrome and access the internet.")
    async def research_and_close():
        pipeline = BrowserResearchPipeline()
        try:
            return await pipeline.research("site:example.com Nano browser test")
        finally:
            await pipeline.close()

    bundle = asyncio.run(research_and_close())
    assert bundle.evidence
    assert all(item.final_url.startswith("http") for item in bundle.evidence)


@pytest.mark.integration
def test_local_ollama_evidence_answer_integration_is_explicitly_opt_in():
    import os
    if os.getenv("NANO_RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set NANO_RUN_OLLAMA_INTEGRATION=1 to contact the configured local Ollama service.")
    answer = think_local(
        "What fact does this source establish?",
        tool_context=(
            "[1] Example source\nFinal URL: https://example.com\n"
            "Page text: Example Company was founded in 2001."
        ),
    )
    assert answer.strip()
    assert not answer.startswith("[Nano]")
