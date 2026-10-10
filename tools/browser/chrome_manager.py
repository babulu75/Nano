from pathlib import Path
from typing import Optional

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from tools.browser.settings import BROWSER_TIMEOUT_MS, CHROME_EXECUTABLE, CHROME_USER_DATA_DIR
from config import debug_log


class BrowserConfigurationError(RuntimeError):
    """Raised when a dedicated Chrome session cannot be configured."""


class ChromeManager:
    # Feature: reusable Chrome session — Nano v0.4 — Purpose: share one browser context across tools and close task tabs separately from app shutdown.
    """Own Nano's persistent Chrome context and provide reusable tab operations."""

    def __init__(
        self,
        executable_path: str = CHROME_EXECUTABLE,
        user_data_dir: str = CHROME_USER_DATA_DIR,
        timeout_ms: int = BROWSER_TIMEOUT_MS,
    ):
        self.executable_path = executable_path
        self.user_data_dir = user_data_dir
        self.timeout_ms = timeout_ms
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None

    async def start(self) -> BrowserContext:
        if self._context is not None:
            debug_log("Reusing active Nano Chrome context.")
            return self._context
        if not self.executable_path or not Path(self.executable_path).is_file():
            raise BrowserConfigurationError(
                "Google Chrome was not found. Set NANO_CHROME_PATH in .env to chrome.exe."
            )
        Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
        debug_log(f"Launching Chrome executable={self.executable_path}; profile={self.user_data_dir}; headless=False; timeout_ms={self.timeout_ms}.")
        self._playwright = await async_playwright().start()
        try:
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir,
                executable_path=self.executable_path,
                headless=False,
                timeout=self.timeout_ms,
                viewport={"width": 1365, "height": 900},
            )
        except Exception:
            debug_log("Chrome launch failed; stopping Playwright runtime.")
            await self._playwright.stop()
            self._playwright = None
            raise
        return self._context

    async def new_page(self) -> Page:
        """Open a task tab in the shared context; callers decide when that tab ends."""
        context = await self.start()
        page = await context.new_page()
        debug_log("Opened a task tab in the shared Chrome session.")
        return page

    async def open_tab(self, url: str) -> Page:
        """Open a reusable browser tab at a URL for browser tools such as Gmail."""
        page = await self.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        except Exception:
            await self.close_page(page)
            raise
        debug_log(f"Opened shared-session tab at {url}.")
        return page

    @staticmethod
    async def close_page(page: Page) -> None:
        """Close a task-owned tab without shutting down the shared browser."""
        try:
            await page.close()
            debug_log("Closed task-owned Chrome tab; shared browser remains available.")
        except Exception as exc:
            debug_log(f"Could not close task-owned Chrome tab: {type(exc).__name__}: {exc}")

    async def close(self) -> None:
        context, playwright = self._context, self._playwright
        self._context = None
        self._playwright = None
        try:
            if context is not None:
                debug_log("Closing shared Nano Chrome context at application shutdown.")
                await context.close()
        finally:
            if playwright is not None:
                debug_log("Stopping Nano-owned Playwright runtime.")
                await playwright.stop()

