from dataclasses import dataclass, replace
from typing import List

from tools.browser.chrome_manager import ChromeManager
from tools.browser.page_reader import PageReadError, PageReader
from tools.browser.search_engine import BrowserSearchEngine, SearchError
from tools.browser.settings import MAX_PAGES, MAX_TOTAL_EVIDENCE_CHARS
from tools.browser.source_validator import EvidenceError, validate_evidence
from models.evidence import Evidence, EvidenceConflict
from config import debug_log


class ResearchError(RuntimeError):
    """Raised when browser research cannot collect usable evidence."""


@dataclass(frozen=True)
class ResearchBundle:
    evidence: List[Evidence]
    conflicts: List[EvidenceConflict]


class BrowserResearchPipeline:
    # Feature: modular browser research pipeline — Nano v0.4 — Purpose: compose search, extraction, validation, and bounded evidence for the local model.
    def __init__(self, chrome_manager=None, search_engine=None, page_reader=None, max_pages=MAX_PAGES, max_total_chars=MAX_TOTAL_EVIDENCE_CHARS):
        self.chrome_manager = chrome_manager or ChromeManager()
        self.search_engine = search_engine or BrowserSearchEngine()
        self.page_reader = page_reader or PageReader()
        self.max_pages = max(1, max_pages)
        self.max_total_chars = max(1, max_total_chars)

    async def research(self, query: str) -> ResearchBundle:
        debug_log(f"Research pipeline started; query_chars={len(query)}; max_pages={self.max_pages}; max_evidence_chars={self.max_total_chars}.")
        evidence = []
        diagnostics = []
        page = None
        try:
            # Feature: reusable browser research tab — Nano v0.4 — Purpose: share Chrome across tools while closing only this search task's tab.
            page = await self.chrome_manager.new_page()
            results = await self.search_engine.search(page, query)
            for result in results[:self.max_pages]:
                try:
                    evidence.append(await self.page_reader.read(page, result))
                except PageReadError as exc:
                    diagnostics.append(str(exc))
                    debug_log(f"Skipping unreadable source: {exc}")
        except SearchError as exc:
            debug_log(f"Research pipeline could not validate evidence: {type(exc).__name__}: {exc}")
            detail = "; ".join(diagnostics)
            suffix = f" Details: {detail}" if detail else ""
            raise ResearchError(f"{exc}{suffix}") from exc
        except Exception as exc:
            debug_log(f"Research pipeline failed unexpectedly: {type(exc).__name__}: {exc}")
            raise ResearchError(f"Browser research failed: {type(exc).__name__}: {exc}") from exc
        finally:
            if page is not None:
                await self.chrome_manager.close_page(page)

        try:
            unique, conflicts = validate_evidence(evidence)
            bounded = []
            remaining = self.max_total_chars
            for item in unique:
                if remaining <= 0:
                    break
                text = item.text[:remaining]
                remaining -= len(text)
                if len(text) < len(item.text):
                    warning = item.warning or f"Total research evidence capped at {self.max_total_chars} characters."
                    item = replace(item, text=text, extraction_status="truncated", warning=warning)
                bounded.append(item)
            if not bounded:
                raise EvidenceError("No evidence text fit within the configured research size limit.")
            debug_log(f"Research pipeline succeeded; sources={len(bounded)}; evidence_chars={sum(len(item.text) for item in bounded)}; conflicts={len(conflicts)}.")
            return ResearchBundle(evidence=bounded, conflicts=conflicts)
        except EvidenceError as exc:
            debug_log(f"Research pipeline could not validate evidence: {type(exc).__name__}: {exc}")
            detail = "; ".join(diagnostics)
            suffix = f" Details: {detail}" if detail else ""
            raise ResearchError(f"{exc}{suffix}") from exc
        except Exception as exc:
            debug_log(f"Research pipeline failed unexpectedly: {type(exc).__name__}: {exc}")
            raise ResearchError(f"Browser research failed: {type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        """Close the shared browser at application shutdown, not after each tool call."""
        await self.chrome_manager.close()


def format_evidence(bundle: ResearchBundle) -> str:
    blocks = []
    for index, item in enumerate(bundle.evidence, start=1):
        publication = item.publication_date or "unknown"
        publication_source = f" (from {item.publication_date_source})" if item.publication_date_source else ""
        warning = f"\nExtraction warning: {item.warning}" if item.warning else ""
        blocks.append(
            f"[{index}] {item.title}\n"
            f"Original result URL: {item.original_url}\n"
            f"Final URL: {item.final_url}\n"
            f"Source domain: {item.source_domain}\n"
            f"Publication/modification date: {publication}{publication_source}\n"
            f"Retrieved at (UTC): {item.retrieved_at}\n"
            f"Page text:\n{item.text}{warning}"
        )
    if bundle.conflicts:
        conflicts = "\n".join(
            f"- Possible conflicting {item.metric} values {', '.join(item.values)} across: {', '.join(item.urls)}"
            for item in bundle.conflicts
        )
        blocks.append("Possible source conflicts detected (verify and describe rather than choosing silently):\n" + conflicts)
    return "\n\n".join(blocks)
