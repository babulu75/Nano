from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit

from tools.browser.search_engine import detect_blocked_page
from tools.browser.settings import BROWSER_RETRIES, BROWSER_TIMEOUT_MS, MAX_EVIDENCE_CHARS
from config import debug_log
from models.evidence import Evidence, SearchResult


class PageReadError(RuntimeError):
    """Raised when a destination page cannot provide readable evidence."""


class PageReader:
    # Feature: rendered webpage evidence extraction — Nano v0.4 — Purpose: read source text and actual page metadata for local answer grounding.
    def __init__(self, timeout_ms: int = BROWSER_TIMEOUT_MS, retries: int = BROWSER_RETRIES, max_chars: int = MAX_EVIDENCE_CHARS):
        self.timeout_ms = timeout_ms
        self.retries = max(0, retries)
        self.max_chars = max(1, max_chars)

    async def read(self, page, result: SearchResult) -> Evidence:
        debug_log(f"Reading source page; url={result.url}; retries={self.retries}.")
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                await page.goto(result.url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.retries and "timeout" in (type(exc).__name__ + str(exc)).lower():
                    debug_log(f"Retrying source navigation after timeout; attempt={attempt + 2}/{self.retries + 1}.")
                    continue
                debug_log(f"Source navigation failed: {type(exc).__name__}: {exc}")
                raise PageReadError(f"Could not open {result.url}: {type(exc).__name__}: {exc}") from exc
        else:
            raise PageReadError(f"Could not open {result.url}: {last_error}")

        blocked_reason = await detect_blocked_page(page)
        if blocked_reason:
            debug_log(f"Source blocked; url={result.url}; reason={blocked_reason}.")
            raise PageReadError(f"Page access was blocked for {result.url}: {blocked_reason}")

        try:
            title = " ".join((await page.title()).split()) or result.title
        except Exception:
            title = result.title
        try:
            body_text = await page.locator("body").inner_text(timeout=self.timeout_ms)
        except Exception as exc:
            raise PageReadError(f"Could not extract readable text from {result.url}: {exc}") from exc
        readable = "\n".join(line.strip() for line in body_text.splitlines() if line.strip())
        if not readable:
            debug_log(f"Source extraction returned no readable body; url={result.url}.")
            raise PageReadError(f"No readable page text was found at {result.url}.")

        publication_date, publication_date_source = await self._publication_date(page)
        truncated = len(readable) > self.max_chars
        text = readable[:self.max_chars]
        final_url = getattr(page, "url", "") or result.url
        domain = urlsplit(final_url).hostname or ""
        debug_log(f"Source extracted; title={title!r}; final_url={final_url}; text_chars={len(text)}; status={'truncated' if truncated else 'ok'}; publication_date={publication_date or 'unknown'}.")
        return Evidence(
            title=title,
            original_url=result.url,
            final_url=final_url,
            text=text,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            source_domain=domain,
            publication_date=publication_date,
            publication_date_source=publication_date_source,
            extraction_status="truncated" if truncated else "ok",
            warning=f"Page text truncated to {self.max_chars} characters." if truncated else None,
        )

    async def _publication_date(self, page) -> tuple[Optional[str], Optional[str]]:
        selectors = (
            ("meta[property='article:published_time']", "content", "article:published_time"),
            ("meta[name='date']", "content", "meta[name=date]"),
            ("meta[name='pubdate']", "content", "meta[name=pubdate]"),
            ("meta[property='og:updated_time']", "content", "og:updated_time"),
            ("time[datetime]", "datetime", "time[datetime]"),
        )
        for selector, attribute, source in selectors:
            try:
                value = await page.locator(selector).first.get_attribute(attribute, timeout=500)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return value.strip(), source
        return None, None
