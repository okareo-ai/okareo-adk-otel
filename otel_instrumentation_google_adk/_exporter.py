"""A fire-and-forget span exporter wrapper.

Wraps any ``SpanExporter`` and submits the real ``export()`` call to a
background thread so that ``SimpleSpanProcessor`` returns immediately
without blocking the agent's hot path on the outbound HTTP request.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class NonBlockingExporter(SpanExporter):
    """Delegates export to a background thread so the caller never blocks."""

    def __init__(self, inner: SpanExporter, max_workers: int = 1) -> None:
        self._inner = inner
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._executor.submit(self._safe_export, spans)
        return SpanExportResult.SUCCESS

    def _safe_export(self, spans: Sequence[ReadableSpan]) -> None:
        try:
            self._inner.export(spans)
        except Exception:
            logger.exception("Background span export failed.")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
