import inspect
import json
import logging
from abc import ABC
from contextlib import ExitStack
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TypeVar,
)

import wrapt
from google.adk import Runner
from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import StatusCode, get_current_span
from opentelemetry.util.types import AttributeValue
from typing_extensions import NotRequired, ParamSpec, TypedDict

from ._helpers import (
    _default,
    bind_args_kwargs,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

P = ParamSpec("P")
T = TypeVar("T")

# ---------------------------------------------------------------------------
# OTEL GenAI semantic convention attribute keys
# ---------------------------------------------------------------------------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
ENDUSER_ID = "enduser.id"

# Operation name values
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

# System value
SYSTEM_GOOGLE_GENAI = "gcp.vertex_ai"


class _WithTracer(ABC):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class _RunnerRunAsyncKwargs(TypedDict):
    user_id: str
    session_id: str
    new_message: types.Content
    run_config: NotRequired[RunConfig]


class _RunnerRunAsync(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Event, None]],
        instance: Runner,
        args: tuple[Any, ...],
        kwargs: _RunnerRunAsyncKwargs,
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        name = f"invocation [{instance.app_name}]"
        attributes: Dict[str, AttributeValue] = {
            GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
            GEN_AI_SYSTEM: SYSTEM_GOOGLE_GENAI,
        }

        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        try:
            attributes["input.value"] = json.dumps(
                arguments,
                default=_default,
                ensure_ascii=False,
            )
            attributes["input.mime_type"] = "application/json"
        except Exception:
            logger.exception("Failed to serialize input value.")

        if (user_id := kwargs.get("user_id")) is not None:
            attributes[ENDUSER_ID] = user_id
        if (session_id := kwargs.get("session_id")) is not None:
            attributes[GEN_AI_CONVERSATION_ID] = session_id

        class _AsyncGenerator(wrapt.ObjectProxy):  # type: ignore[misc]
            __wrapped__: AsyncGenerator[Event, None]

            async def __aiter__(self) -> Any:
                with ExitStack() as stack:
                    span = stack.enter_context(
                        tracer.start_as_current_span(
                            name=name,
                            attributes=attributes,
                        )
                    )
                    async for event in self.__wrapped__:
                        if event.is_final_response():
                            try:
                                span.set_attribute(
                                    "output.value",
                                    event.model_dump_json(exclude_none=True),
                                )
                                span.set_attribute(
                                    "output.mime_type",
                                    "application/json",
                                )
                            except Exception:
                                logger.exception("Failed to serialize output value.")
                        yield event
                    span.set_status(StatusCode.OK)

        return _AsyncGenerator(generator)


class _BaseAgentRunAsync(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Event, None]],
        instance: BaseAgent,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        name = f"agent_run [{instance.name}]"
        attributes: Dict[str, AttributeValue] = {
            GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
            GEN_AI_SYSTEM: SYSTEM_GOOGLE_GENAI,
            GEN_AI_AGENT_NAME: instance.name,
        }

        class _AsyncGenerator(wrapt.ObjectProxy):  # type: ignore[misc]
            __wrapped__: AsyncGenerator[Event, None]

            async def __aiter__(self) -> Any:
                with tracer.start_as_current_span(
                    name=name,
                    attributes=attributes,
                ) as span:
                    async for event in self.__wrapped__:
                        if event.is_final_response():
                            try:
                                span.set_attribute(
                                    "output.value",
                                    event.model_dump_json(exclude_none=True),
                                )
                                span.set_attribute(
                                    "output.mime_type",
                                    "application/json",
                                )
                            except Exception:
                                logger.exception("Failed to serialize output value.")
                        yield event
                    span.set_status(StatusCode.OK)

        return _AsyncGenerator(generator)


class _TraceCallLlm(_WithTracer):
    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., T],
        _: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> T:
        ans = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return ans
        span = get_current_span()
        span.set_status(StatusCode.OK)
        span.set_attribute(GEN_AI_OPERATION_NAME, OP_CHAT)
        span.set_attribute(GEN_AI_SYSTEM, SYSTEM_GOOGLE_GENAI)

        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        llm_request = next((arg for arg in arguments.values() if isinstance(arg, LlmRequest)), None)
        llm_response = next(
            (arg for arg in arguments.values() if isinstance(arg, LlmResponse)), None
        )

        if llm_request:
            if llm_request.model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, llm_request.model)

            if config := llm_request.config:
                for k, v in _get_attributes_from_generate_content_config(config):
                    span.set_attribute(k, v)

            # Build input messages as JSON array
            input_messages = _build_input_messages(llm_request)
            if input_messages:
                try:
                    span.set_attribute(
                        GEN_AI_INPUT_MESSAGES,
                        json.dumps(input_messages, default=_default, ensure_ascii=False),
                    )
                except Exception:
                    logger.exception("Failed to serialize input messages.")

            # Tool definitions
            if llm_request.tools_dict:
                for k, v in _get_attributes_from_tools(llm_request.tools_dict):
                    span.set_attribute(k, v)

        if llm_response:
            for k, v in _get_attributes_from_llm_response(llm_response):
                span.set_attribute(k, v)

        return ans


class _TraceToolCall(_WithTracer):
    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., T],
        _: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> T:
        ans = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return ans
        span = get_current_span()
        span.set_status(StatusCode.OK)
        span.set_attribute(GEN_AI_OPERATION_NAME, OP_EXECUTE_TOOL)
        span.set_attribute(GEN_AI_SYSTEM, SYSTEM_GOOGLE_GENAI)

        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        if base_tool := next(
            (arg for arg in arguments.values() if isinstance(arg, BaseTool)), None
        ):
            span.set_attribute(GEN_AI_TOOL_NAME, base_tool.name)
            if base_tool.description:
                span.set_attribute(GEN_AI_TOOL_DESCRIPTION, base_tool.description)
            if args_dict := next(
                (arg for arg in arguments.values() if isinstance(arg, Mapping)), None
            ):
                try:
                    span.set_attribute("input.value", safe_json_dumps(args_dict))
                    span.set_attribute("input.mime_type", "application/json")
                except Exception:
                    logger.exception("Failed to serialize tool input.")
        if event := next((arg for arg in arguments.values() if isinstance(arg, Event)), None):
            if responses := event.get_function_responses():
                try:
                    span.set_attribute(
                        "output.value",
                        responses[0].model_dump_json(exclude_none=True),
                    )
                    span.set_attribute("output.mime_type", "application/json")
                except Exception:
                    logger.exception("Failed to serialize tool output.")
        return ans


# ---------------------------------------------------------------------------
# Attribute extraction helpers
# ---------------------------------------------------------------------------


def stop_on_exception(
    wrapped: Callable[P, Iterator[tuple[str, AttributeValue]]],
) -> Callable[P, Iterator[tuple[str, AttributeValue]]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[tuple[str, AttributeValue]]:
        try:
            yield from wrapped(*args, **kwargs)
        except Exception:
            logger.exception(f"Failed to get attribute in {wrapped.__name__}.")

    return wrapper


@stop_on_exception
def _get_attributes_from_generate_content_config(
    obj: types.GenerateContentConfig,
) -> Iterator[tuple[str, AttributeValue]]:
    if obj.temperature is not None:
        yield GEN_AI_REQUEST_TEMPERATURE, obj.temperature
    if obj.top_p is not None:
        yield GEN_AI_REQUEST_TOP_P, obj.top_p


@stop_on_exception
def _get_attributes_from_llm_response(
    obj: LlmResponse,
) -> Iterator[tuple[str, AttributeValue]]:
    if obj.usage_metadata:
        yield from _get_attributes_from_usage_metadata(obj.usage_metadata)
    if obj.content:
        output_messages = [_content_to_message_dict(obj.content)]
        try:
            yield (
                GEN_AI_OUTPUT_MESSAGES,
                json.dumps(output_messages, default=_default, ensure_ascii=False),
            )
        except Exception:
            logger.exception("Failed to serialize output messages.")


@stop_on_exception
def _get_attributes_from_usage_metadata(
    obj: types.GenerateContentResponseUsageMetadata,
) -> Iterator[tuple[str, AttributeValue]]:
    if prompt := obj.prompt_token_count:
        yield GEN_AI_USAGE_INPUT_TOKENS, prompt
    completion = 0
    if candidates := obj.candidates_token_count:
        completion += candidates
    if thoughts := obj.thoughts_token_count:
        completion += thoughts
    if completion:
        yield GEN_AI_USAGE_OUTPUT_TOKENS, completion


@stop_on_exception
def _get_attributes_from_tools(
    tools_dict: Mapping[str, BaseTool],
) -> Iterator[tuple[str, AttributeValue]]:
    tool_defs: List[Dict[str, Any]] = []
    for tool in tools_dict.values():
        if declaration := tool._get_declaration():
            tool_defs.append(json.loads(declaration.model_dump_json(exclude_none=True)))
        else:
            tool_defs.append({"name": tool.name, "description": tool.description})
    if tool_defs:
        yield "gen_ai.request.tools", json.dumps(tool_defs, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Message building — produces dicts that get JSON-serialized into
# gen_ai.input.messages / gen_ai.output.messages
# ---------------------------------------------------------------------------


def _build_input_messages(llm_request: LlmRequest) -> List[Dict[str, Any]]:
    """Build the list of message dicts from an LlmRequest."""
    messages: List[Dict[str, Any]] = []

    if config := llm_request.config:
        if system_instruction := config.system_instruction:
            msg = _system_instruction_to_message_dict(system_instruction)
            if msg:
                messages.append(msg)

    if contents := llm_request.contents:
        for content in contents:
            messages.append(_content_to_message_dict(content))

    return messages


def _system_instruction_to_message_dict(
    system_instruction: Any,
) -> Optional[Dict[str, Any]]:
    """Convert a system instruction (str, Content, or list) to a message dict."""
    if isinstance(system_instruction, str):
        return {"role": "system", "content": system_instruction}
    elif isinstance(system_instruction, types.Content):
        if system_instruction.parts:
            texts = [p.text for p in system_instruction.parts if p.text is not None]
            if texts:
                return {"role": "system", "content": "\n".join(texts)}
    return None


def _content_to_message_dict(obj: types.Content) -> Dict[str, Any]:
    """Convert a google.genai Content object to a GenAI message dict."""
    role = obj.role or "user"
    msg: Dict[str, Any] = {"role": role}

    if not obj.parts:
        return msg

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []

    for part in obj.parts:
        if part.text is not None:
            text_parts.append(part.text)
        elif (fc := part.function_call) is not None:
            tc: Dict[str, Any] = {"type": "function", "function": {}}
            if fc.id:
                tc["id"] = fc.id
            if fc.name:
                tc["function"]["name"] = fc.name
            if fc.args:
                tc["function"]["arguments"] = safe_json_dumps(fc.args)
            tool_calls.append(tc)
        elif (fr := part.function_response) is not None:
            tr: Dict[str, Any] = {"role": "tool"}
            if fr.name:
                tr["name"] = fr.name
            if fr.response:
                tr["content"] = safe_json_dumps(fr.response)
            tool_results.append(tr)

    # If we have function responses, they become separate tool-role messages.
    # But if mixed with other parts in the same Content, we still represent
    # the primary message and return tool results via the caller.
    if text_parts:
        msg["content"] = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if tool_results:
        # function_response parts in a Content indicate tool-role messages.
        # When mixed into a single Content with other parts (rare), we
        # flatten — the first tool result replaces role/content.
        if not text_parts and not tool_calls:
            # Pure tool-response content
            first = tool_results[0]
            msg["role"] = "tool"
            if "name" in first:
                msg["name"] = first["name"]
            if "content" in first:
                msg["content"] = first["content"]

    return msg
