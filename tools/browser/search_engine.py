import asyncio
from typing import List, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlsplit, urlunsplit

from models.evidence import SearchResult
from tools.browser.settings import BROWSER_RETRIES, BROWSER_TIMEOUT_MS, MAX_SEARCH_RESULTS, SEARCH_ENGINES
from config import debug_log


class SearchError(RuntimeError):
    """Base error for browser search failures."""


class SearchBlockedError(SearchError):
    """Raised when an engine blocks automation or requires human verification."""


class SearchTimeoutError(SearchError):
    """Raised after the configured bounded navigation retries time out."""


class NoSearchResultsError(SearchError):
    """Raised when all supported pages have no extractable result links."""


SEARCH_HOSTS = {"google.com", "bing.com", "duckduckgo.com", "microsoft.com"}
INVALID_PATH_PARTS = ("/settings", "/preferences", "/account", "/search", "/sorry")
BLOCK_MARKERS = (
    "unusual traffic",
    "verify you are human",
    "verify that you are human",
    "captcha",
    "access denied",
    "automated queries",
    "your request has been blocked",
    "not a robot",
)


def valid_destination(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in {"http", "https"} or not host or "." not in host:
            return False
        if any(host == domain or host.endswith("." + domain) for domain in SEARCH_HOSTS):
            return False
        if any(part in parsed.path.lower() for part in INVALID_PATH_PARTS):
            return False
        return True
    except (TypeError, ValueError):
        return False


def normalize_destination(href: str, base_url: str) -> Optional[str]:
    if not isinstance(href, str) or not href.strip():
        return None
    url = urljoin(base_url, href.strip())
    parsed = urlsplit(url)
    if "bing.com" in (parsed.hostname or "").lower() and parsed.path.startswith("/ck/a"):
        encoded = parse_qs(parsed.query).get("u", [""])[0]
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        if encoded:
            url = unquote(encoded)
    parsed = urlsplit(url)
    url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    return url if valid_destination(url) else None


async def detect_blocked_page(page) -> Optional[str]:
    try:
        title = (await page.title()).lower()
    except Exception:
        title = ""
    try:
        body = (await page.locator("body").inner_text(timeout=1500)).lower()
    except Exception:
        body = ""
    url = (getattr(page, "url", "") or "").lower()
    if "/sorry/" in url:
        return "search engine CAPTCHA or unusual-traffic check"
    content = f"{title}\n{body}"
    marker = next((text for text in BLOCK_MARKERS if text in content), None)
    return f"search engine reported '{marker}'" if marker else None


async def extract_search_results(page, engine: str, limit: int = MAX_SEARCH_RESULTS) -> List[SearchResult]:
    # Feature: browser search-result extraction — Nano v0.4 — Purpose: extract bounded, deduplicated external result links from rendered search pages.
    selectors = {
        "Google": ("a:has(h3)",),
        "Bing": ("li.b_algo h2 a", "li.b_algo a[href]"),
        "DuckDuckGo": ("a.result__a",),
    }.get(engine, ())
    candidates = []
    for selector in (*selectors, "a[href]"):
        try:
            links = page.locator(selector)
            count = await links.count()
        except Exception:
            continue
        for index in range(min(count, 250)):
            link = links.nth(index)
            try:
                title = " ".join((await link.inner_text(timeout=800)).split())
                href = await link.get_attribute("href")
            except Exception:
                continue
            destination = normalize_destination(href, getattr(page, "url", ""))
            if not title or len(title) < 3 or len(title) > 250 or not destination:
                continue
            if title.lower() in {"images", "videos", "maps", "news", "shopping", "more", "sign in", "settings"}:
                continue
            candidates.append(SearchResult(title=title[:200], url=destination, engine=engine))
            if len(candidates) >= limit * 2:
                break
        if candidates:
            break

    unique = []
    seen = set()
    for result in candidates:
        key = normalize_destination(result.url, result.url)
        if key and key not in seen:
            seen.add(key)
            unique.append(result)
        if len(unique) >= limit:
            break
    debug_log(f"Search extraction complete; engine={engine}; candidates={len(candidates)}; accepted={len(unique)}.")
    return unique


class BrowserSearchEngine:
    def __init__(self, engines=SEARCH_ENGINES, timeout_ms=BROWSER_TIMEOUT_MS, retries=BROWSER_RETRIES, result_limit=MAX_SEARCH_RESULTS):
        self.engines = engines
        self.timeout_ms = timeout_ms
        self.retries = max(0, retries)
        self.result_limit = max(1, result_limit)

    async def search(self, page, query: str) -> List[SearchResult]:
        # Feature: multi-engine browser search with safe fallback — Nano v0.4 — Purpose: retrieve web sources while stopping blocked engine attempts.
        debug_log(f"Search requested; query_chars={len(query)}; engines={[name for name, _ in self.engines]}; retries={self.retries}.")
        errors = []
        blocked_engines = []
        last_timeout = False
        for engine, template in self.engines:
            url = template.format(query=quote_plus(query))
            for attempt in range(self.retries + 1):
                try:
                    debug_log(f"Navigating search engine={engine}; attempt={attempt + 1}/{self.retries + 1}; url={url}.")
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                except Exception as exc:
                    if "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower():
                        last_timeout = True
                        errors.append(f"{engine} navigation timed out")
                        if attempt < self.retries:
                            debug_log(f"Retrying {engine} after navigation timeout.")
                            await asyncio.sleep(min(0.5 * (attempt + 1), 1.0))
                            continue
                        break
                    errors.append(f"{engine} navigation failed: {type(exc).__name__}")
                    debug_log(f"Search navigation failed for {engine}: {type(exc).__name__}: {exc}")
                    break

                reason = await detect_blocked_page(page)
                if reason:
                    message = f"{engine} blocked the search ({reason}); stopping that engine attempt."
                    debug_log(message)
                    errors.append(message)
                    blocked_engines.append(engine)
                    break
                selector = {
                    "Google": "a:has(h3)",
                    "Bing": "li.b_algo h2 a",
                    "DuckDuckGo": "a.result__a",
                }.get(engine)
                if selector:
                    try:
                        await page.locator(selector).first.wait_for(state="attached", timeout=1500)
                    except Exception:
                        pass
                results = await extract_search_results(page, engine, self.result_limit)
                if results:
                    for index, result in enumerate(results, start=1):
                        debug_log(f"Search result {index}: {result.title} | {result.url}")
                    debug_log(f"Search succeeded; engine={engine}; result_count={len(results)}.")
                    return results
                errors.append(f"{engine} page loaded but no result links were extractable")
                debug_log(f"No result links extracted from {engine}; trying the next engine if available.")
                break

        if last_timeout:
            raise SearchTimeoutError("Search navigation timed out after bounded retries: " + "; ".join(errors))
        if blocked_engines and len(blocked_engines) == len(self.engines):
            raise SearchBlockedError("All configured search engines blocked automated access: " + "; ".join(errors))
        raise NoSearchResultsError("No extractable results from supported search pages: " + "; ".join(errors))
