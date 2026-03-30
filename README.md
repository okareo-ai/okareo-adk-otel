# otel-instrumentation-google-adk

OpenTelemetry instrumentation for Google ADK (Agent Development Kit).

This instrumentor patches the Google ADK runner, agent, LLM, and tool layers to produce spans that follow the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (`gen_ai.*` attributes). It is designed to work inside Google Agent Engine (GAE) where other telemetry libraries may be stripped.

## Installation

```bash
pip install otel-instrumentation-google-adk
```

## Usage

```python
from otel_instrumentation_google_adk import GoogleADKInstrumentor

GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
```

## What gets instrumented

- **Runner.run_async** — creates an `invocation` span with input/output values, user ID, and session ID
- **BaseAgent.run_async** — creates an `agent_run` span per agent with agent name and output
- **LLM calls** — enriches spans with model name, input/output messages, token usage, temperature, and tool definitions
- **Tool calls** — enriches spans with tool name, description, input arguments, and output

## Span attributes

Spans follow the OpenTelemetry GenAI semantic conventions:

| Attribute | Description |
|---|---|
| `gen_ai.operation.name` | `invoke_agent`, `chat`, or `execute_tool` |
| `gen_ai.system` | `gcp.vertex_ai` |
| `gen_ai.request.model` | Model name from the LLM request |
| `gen_ai.agent.name` | Agent name |
| `gen_ai.input.messages` | JSON-serialized input messages |
| `gen_ai.output.messages` | JSON-serialized output messages |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count (candidates + thoughts) |
| `gen_ai.tool.name` | Tool name |
| `gen_ai.request.tools` | JSON-serialized tool definitions |

## License

Apache-2.0
