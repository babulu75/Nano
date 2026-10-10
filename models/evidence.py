from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""


@dataclass(frozen=True)
class Evidence:
    title: str
    original_url: str
    final_url: str
    text: str
    retrieved_at: str
    source_domain: str
    publication_date: Optional[str] = None
    publication_date_source: Optional[str] = None
    extraction_status: str = "ok"
    warning: Optional[str] = None


@dataclass(frozen=True)
class EvidenceConflict:
    metric: str
    values: tuple[str, ...]
    urls: tuple[str, ...]
