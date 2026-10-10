import re
from typing import Iterable, List, Tuple
from urllib.parse import urlsplit, urlunsplit

from models.evidence import Evidence, EvidenceConflict
from config import debug_log


class EvidenceError(RuntimeError):
    """Raised when collected evidence cannot safely support an answer."""


METRIC_PATTERNS = {
    "price": re.compile(r"\b(?:price|cost|trading at)\b[^\d$€£₹]{0,24}([$€£₹]?\s?\d[\d,.]*(?:\.\d+)?\s?(?:%|million|billion|thousand)?)", re.I),
    "revenue": re.compile(r"\brevenue\b[^\d$€£₹]{0,24}([$€£₹]?\s?\d[\d,.]*(?:\.\d+)?\s?(?:%|million|billion|thousand)?)", re.I),
    "population": re.compile(r"\bpopulation\b[^\d]{0,24}(\d[\d,.]*(?:\.\d+)?\s?(?:million|billion|thousand)?)", re.I),
    "score": re.compile(r"\bscore\b[^\d]{0,24}(\d[\d,.]*(?:\s?[-:]\s?\d[\d,.]*)?)", re.I),
    "percentage": re.compile(r"\b(?:rate|percentage|percent)\b[^\d]{0,24}(\d[\d,.]*\s?%)", re.I),
}


def canonical_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, ""))


def validate_evidence(evidence: Iterable[Evidence]) -> Tuple[List[Evidence], List[EvidenceConflict]]:
    # Feature: evidence validation — Nano v0.4 — Purpose: reject empty reads, deduplicate sources, and expose conflicting observed metrics.
    unique = []
    seen = set()
    for item in evidence:
        if item.extraction_status not in {"ok", "truncated"} or not item.text.strip():
            continue
        key = canonical_url(item.final_url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    if not unique:
        debug_log("Evidence validation failed: no unique readable source pages.")
        raise EvidenceError("No readable source pages were available to verify this request.")
    conflicts = find_conflicts(unique)
    debug_log(f"Evidence validation succeeded; unique_sources={len(unique)}; conflicts={len(conflicts)}.")
    return unique, conflicts


def find_conflicts(evidence: Iterable[Evidence]) -> List[EvidenceConflict]:
    observations = {}
    for item in evidence:
        for metric, pattern in METRIC_PATTERNS.items():
            values = {" ".join(value.split()).lower().rstrip(".,;:") for value in pattern.findall(item.text)}
            for value in values:
                observations.setdefault((metric, value), set()).add(item.final_url)

    conflicts = []
    for metric in METRIC_PATTERNS:
        values = [(value, urls) for (observed_metric, value), urls in observations.items() if observed_metric == metric]
        unique_sources = {url for _, urls in values for url in urls}
        if len(values) > 1 and len(unique_sources) > 1:
            conflicts.append(EvidenceConflict(
                metric=metric,
                values=tuple(sorted(value for value, _ in values)),
                urls=tuple(sorted({url for _, urls in values for url in urls})),
            ))
    return conflicts
