from dataclasses import dataclass
from typing import List
from urllib.parse import parse_qs, unquote, urlsplit

from tools.browser.settings import BROWSER_TIMEOUT_MS
from config import debug_log


class GoogleAiModeError(RuntimeError):
    """Raised when Google AI Mode cannot provide a copied answer."""


@dataclass(frozen=True)
class GoogleAiAnswer:
    text: str
    page_url: str
    citations: List[str]


class GoogleAiMode:
    """Ask Google AI Mode in the visible browser and copy its answer text."""

    URL = "https://www.google.com/search?udm=50&aep=11"
    ANSWER_TIMEOUT_MS = max(BROWSER_TIMEOUT_MS, 90_000)

    async def ask(self, page, query: str) -> GoogleAiAnswer:
        await page.context.grant_permissions(
            ["clipboard-read", "clipboard-write"],
            origin="https://www.google.com",
        )
        debug_log(f"Opening Google AI Mode; query_chars={len(query)}.")
        await page.goto(self.URL, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT_MS)

        composer = await self._find_composer(page)
        if composer is None:
            raise GoogleAiModeError(
                "Google AI Mode's question box was not found. Check that AI Mode is available and signed in in Nano's Chrome profile."
            )
        await composer.fill(query)
        await composer.press("Enter")
        debug_log("Submitted query to Google AI Mode; waiting for its Copy answer control.")

        copy_button = page.locator(
            "button[aria-label*='copy' i], button[title*='copy' i], "
            "[role='button'][aria-label*='copy' i], [role='button'][title*='copy' i]"
        ).last
        try:
            await copy_button.wait_for(state="visible", timeout=self.ANSWER_TIMEOUT_MS)
        except Exception as exc:
            raise GoogleAiModeError(
                "Google AI Mode did not show a copy-answer button before the timeout. "
                "It may still be generating, unavailable for this account, or asking for sign-in."
            ) from exc

        page_url = page.url
        citations = await self._visible_source_links(page)
        try:
            await copy_button.click()
            await page.wait_for_timeout(150)
            copied_text = await page.evaluate("() => navigator.clipboard.readText()")
        except Exception as exc:
            raise GoogleAiModeError(
                f"Google AI Mode displayed a copy button, but Nano could not read its copied answer: {exc}"
            ) from exc

        answer = copied_text.strip() if isinstance(copied_text, str) else ""
        if not answer or answer == query.strip():
            raise GoogleAiModeError("Google AI Mode's copy action returned no answer text.")
        debug_log(f"Copied Google AI Mode answer; answer_chars={len(answer)}; citations={len(citations)}.")
        return GoogleAiAnswer(text=answer, page_url=page_url, citations=citations)

    async def _find_composer(self, page):
        selectors = (
            "textarea[aria-label*='ask' i]",
            "textarea[placeholder*='ask' i]",
            "[contenteditable='true'][aria-label*='ask' i]",
            "[role='textbox'][aria-label*='ask' i]",
            "textarea",
            "[contenteditable='true'][role='textbox']",
            "[contenteditable='true']",
            "[role='textbox']",
            "input[type='text']",
        )
        for selector in selectors:
            fields = page.locator(selector)
            for index in range(await fields.count()):
                candidate = fields.nth(index)
                if await candidate.is_visible():
                    return candidate
        return None

    async def _visible_source_links(self, page) -> List[str]:
        links = page.locator("a[href^='http']")
        found = []
        for index in range(await links.count()):
            link = links.nth(index)
            try:
                if not await link.is_visible():
                    continue
                href = await link.get_attribute("href")
                href = self._unwrap_google_link(href or "")
                host = urlsplit(href).hostname or ""
                if host and host.lower().removeprefix("www.") not in {
                    "google.com", "accounts.google.com"
                }:
                    found.append(href)
            except Exception:
                continue
        # Keep the visible citation set bounded and unique while preserving order.
        unique = list(dict.fromkeys(found))
        return unique[:12]

    @staticmethod
    def _unwrap_google_link(href: str) -> str:
        parsed = urlsplit(href)
        if (parsed.hostname or "").lower().removeprefix("www.") == "google.com":
            query = parse_qs(parsed.query)
            for key in ("q", "url", "target"):
                candidate = query.get(key, [""])[0]
                if candidate.startswith(("http://", "https://")):
                    return unquote(candidate)
        return href
