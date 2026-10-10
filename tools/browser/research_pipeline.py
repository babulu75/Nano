from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlsplit

from tools.browser.chrome_manager import ChromeManager
from tools.browser.google_ai_mode import GoogleAiMode, GoogleAiModeError
from tools.browser.settings import MAX_TOTAL_EVIDENCE_CHARS
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
    # Feature: Google AI Mode research — Nano v0.4 — Purpose: use Google's visible AI answer and citations as evidence for Nano's concise local response.
    def __init__(self, chrome_manager=None, ai_mode=None, max_total_chars=MAX_TOTAL_EVIDENCE_CHARS):
        self.chrome_manager = chrome_manager or ChromeManager()
        self.ai_mode = ai_mode or GoogleAiMode()
        self.max_total_chars = max(1, max_total_chars)

    async def research(self, query: str) -> ResearchBundle:
        debug_log(f"Research pipeline started; query_chars={len(query)}; max_evidence_chars={self.max_total_chars}.")
        page = None
        try:
            page = await self.chrome_manager.new_page()
            answer = await self.ai_mode.ask(page, query)
            result_text = answer.text
            if answer.citations:
                result_text += "\n\nSources shown by Google AI Mode:\n" + "\n".join(answer.citations)
            evidence = [Evidence(
                title=f"Google AI Mode answer for: {query}",
                original_url=answer.page_url,
                final_url=answer.page_url,
                text=str(result_text)[:self.max_total_chars],
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                source_domain=urlsplit(answer.page_url).hostname or "",
                extraction_status="ok",
            )]
        except GoogleAiModeError as exc:
            debug_log(f"Research pipeline could not validate evidence: {type(exc).__name__}: {exc}")
            raise ResearchError(str(exc)) from exc
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
            raise ResearchError(str(exc)) from exc
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
            f"Google AI Mode answer and citations (untrusted page content):\n{item.text}{warning}"
        )
    if bundle.conflicts:
        conflicts = "\n".join(
            f"- Possible conflicting {item.metric} values {', '.join(item.values)} across: {', '.join(item.urls)}"
            for item in bundle.conflicts
        )
        blocks.append("Possible source conflicts detected (verify and describe rather than choosing silently):\n" + conflicts)
    return "\n\n".join(blocks)
