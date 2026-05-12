import logging
from typing import Any, Collection, Dict, Iterator, List, Tuple, cast

import wrapt
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from opentelemetry.trace import Span, Tracer, get_current_span
from opentelemetry.util._decorator import _agnosticcontextmanager
from wrapt import resolve_path, wrap_function_wrapper

from ._exporter import NonBlockingExporter
from .version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("google-adk >= 1.2.1",)


class GoogleADKInstrumentor(BaseInstrumentor):  # type: ignore
    """Instruments Google ADK to emit standard OpenTelemetry GenAI spans.

    This instrumentor patches the Google ADK runner, agent, LLM, and tool
    layers to produce spans that follow the OpenTelemetry GenAI semantic
    conventions (``gen_ai.*`` attributes).  It is designed to work inside
    Google Agent Engine (GAE) where other telemetry libraries are stripped.

    Usage::

        from otel_instrumentation_google_adk import GoogleADKInstrumentor

        GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()

        self._tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        from google.adk.agents import BaseAgent
        from google.adk.runners import Runner

        from ._wrappers import (
            _BaseAgentRunAsync,
            _RunnerRunAsync,
        )

        # Store original methods for cleanup during uninstrumentation
        self._originals: List[Tuple[Any, Any, Any]] = []
        method_wrappers: Dict[Any, Any] = {
            Runner.run_async: _RunnerRunAsync(self._tracer),
            BaseAgent.run_async: _BaseAgentRunAsync(self._tracer),
        }

        # Wrap each method with its corresponding tracer
        for method, wrapper in method_wrappers.items():
            module, name = method.__module__, method.__qualname__
            self._originals.append(resolve_path(module, name))
            wrap_function_wrapper(module, name, wrapper)

        self._patch_trace_call_llm()
        self._patch_trace_tool_call()
        self._disable_existing_tracers()

    def _uninstrument(self, **kwargs: Any) -> None:
        self._unpatch_trace_call_llm()
        self._unpatch_trace_tool_call()
        self._restore_existing_tracers()

        # Restore all wrapped methods to their original state
        for parent, attribute, original in getattr(self, "_originals", ()):
            setattr(parent, attribute, original)

    def _patch_trace_call_llm(self) -> None:
        from google.adk.flows.llm_flows import base_llm_flow

        from ._wrappers import _TraceCallLlm

        original = getattr(base_llm_flow, "trace_call_llm", None)
        if original is None:
            logger.debug(
                "google.adk.flows.llm_flows.base_llm_flow.trace_call_llm not found; "
                "skipping trace_call_llm instrumentation",
            )
            return

        setattr(base_llm_flow, "tracer", self._tracer)
        setattr(base_llm_flow, "trace_call_llm", _TraceCallLlm(self._tracer)(original))

    def _unpatch_trace_call_llm(self) -> None:
        from google.adk.flows.llm_flows import base_llm_flow

        current = getattr(base_llm_flow, "trace_call_llm", None)
        if current is None:
            return

        if callable(original := getattr(current, "__wrapped__", None)):
            setattr(base_llm_flow, "trace_call_llm", original)

        try:
            from google.adk.telemetry import tracer
        except ImportError:
            return

        setattr(base_llm_flow, "tracer", tracer)

    def _patch_trace_tool_call(self) -> None:
        from google.adk.flows.llm_flows import functions

        from ._wrappers import _TraceToolCall

        original = getattr(functions, "trace_tool_call", None)
        if original is None:
            logger.debug(
                "google.adk.flows.llm_flows.functions.trace_tool_call not found; "
                "skipping trace_tool_call instrumentation",
            )
            return

        setattr(functions, "tracer", self._tracer)
        setattr(functions, "trace_tool_call", _TraceToolCall(self._tracer)(original))

    def _unpatch_trace_tool_call(self) -> None:
        from google.adk.flows.llm_flows import functions

        current = getattr(functions, "trace_tool_call", None)
        if current is None:
            return

        if callable(original := getattr(current, "__wrapped__", None)):
            setattr(functions, "trace_tool_call", original)

        try:
            from google.adk.telemetry import tracer
        except ImportError:
            return

        setattr(functions, "tracer", tracer)

    def _disable_existing_tracers(self) -> None:
        """Disable existing tracers to prevent double-instrumentation."""
        from google.adk import runners

        runners_tracer = getattr(runners, "tracer", None)
        if isinstance(runners_tracer, Tracer):
            setattr(runners, "tracer", _PassthroughTracer(runners_tracer))

        from google.adk.agents import base_agent

        base_agent_tracer = getattr(base_agent, "tracer", None)
        if isinstance(base_agent_tracer, Tracer):
            setattr(base_agent, "tracer", _PassthroughTracer(base_agent_tracer))

        from google.adk import __version__

        version = cast(tuple[int, int, int], tuple(int(x) for x in __version__.split(".")[:3]))

        if version >= (1, 15, 0):
            try:
                from google.adk.telemetry import (  # type: ignore[attr-defined,import-not-found,unused-ignore]
                    tracing as adk_tracing,  # type: ignore[attr-defined,unused-ignore]
                )
            except ImportError:
                logger.debug("google.adk.telemetry.tracing not found; skipping")
                return

            adk_tracing_tracer = getattr(adk_tracing, "tracer", None)
            if isinstance(adk_tracing_tracer, Tracer):
                setattr(adk_tracing, "tracer", _PassthroughTracer(adk_tracing_tracer))

    def _restore_existing_tracers(self) -> None:
        """Restore original tracers that were disabled during instrumentation."""
        from google.adk import runners

        runners_tracer = getattr(runners, "tracer", None)
        if runners_tracer is not None and isinstance(
            original := getattr(runners_tracer, "__wrapped__", None), Tracer
        ):
            setattr(runners, "tracer", original)

        from google.adk.agents import base_agent

        base_agent_tracer = getattr(base_agent, "tracer", None)
        if base_agent_tracer is not None and isinstance(
            original := getattr(base_agent_tracer, "__wrapped__", None), Tracer
        ):
            setattr(base_agent, "tracer", original)

        from google.adk import __version__

        version = cast(tuple[int, int, int], tuple(int(x) for x in __version__.split(".")[:3]))

        if version >= (1, 15, 0):
            try:
                from google.adk.telemetry import (  # type: ignore[attr-defined,import-not-found,unused-ignore]
                    tracing as adk_tracing,  # type: ignore[attr-defined,unused-ignore]
                )
            except ImportError:
                return

            adk_tracing_tracer = getattr(adk_tracing, "tracer", None)
            if adk_tracing_tracer is not None and isinstance(
                original := getattr(adk_tracing_tracer, "__wrapped__", None), Tracer
            ):
                setattr(adk_tracing, "tracer", original)


class _PassthroughTracer(wrapt.ObjectProxy):  # type: ignore[misc]
    """A tracer proxy that passes through span operations without creating new spans.

    This is used to disable existing tracers during instrumentation to prevent
    double-instrumentation of the same operations.
    """

    @_agnosticcontextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        """Return the current span without creating a new one."""
        yield get_current_span()
