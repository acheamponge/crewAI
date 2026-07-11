"""Typed contexts for the interception points wired in phases 2-5.

Each context is a dataclass whose fields are nullable and defaulted, so a field
that is not meaningful for a given runtime (e.g. ``agent_role`` inside a flow)
is simply ``None`` rather than an error. Every context exposes a ``payload``
field: the interceptable value a hook may mutate in place or replace by
returning a new value.

The legacy ``pre/post_model_call`` and ``pre/post_tool_call`` points keep using
:class:`~crewai.hooks.llm_hooks.LLMCallHookContext` and
:class:`~crewai.hooks.tool_hooks.ToolCallHookContext` for backwards
compatibility; they are intentionally not redefined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InterceptionContext:
    """Base context shared by the framework-native interception points."""

    payload: Any = None
    agent: Any = None
    agent_role: str | None = None
    task: Any = None
    crew: Any = None
    flow: Any = None


@dataclass
class ExecutionStartContext(InterceptionContext):
    """``execution_start``: a crew or flow is about to begin. ``payload`` = inputs."""

    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputContext(InterceptionContext):
    """``input``: resolved inputs for an execution. ``payload`` = inputs."""

    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputContext(InterceptionContext):
    """``output``: final result of a crew or flow. ``payload`` = the output object."""

    output: Any = None


@dataclass
class ExecutionEndContext(InterceptionContext):
    """``execution_end``: a crew or flow has finished. ``payload`` = the output object."""

    output: Any = None


@dataclass
class StepContext(InterceptionContext):
    """``pre_step`` / ``post_step``: a task or flow-method step boundary.

    ``kind`` is ``"task"`` for crew tasks and ``"flow_method"`` for flow methods.
    ``payload`` is the step input (pre) or step output (post).
    """

    kind: str | None = None
    step_name: str | None = None
    output: Any = None


@dataclass
class ToolSelectionContext(InterceptionContext):
    """``tool_selection``: the set of tools offered to an agent. ``payload`` = tools list."""

    tools: list[Any] = field(default_factory=list)


@dataclass
class PreDelegationContext(InterceptionContext):
    """``pre_delegation``: an agent is about to delegate work. ``payload`` = delegation input."""

    coworker: str | None = None
    delegate_to: Any = None


@dataclass
class RetryAttemptContext(InterceptionContext):
    """``retry_attempt``: an operation is about to be retried."""

    attempt: int = 0
    max_attempts: int | None = None
    error: Any = None


@dataclass
class MemoryWriteContext(InterceptionContext):
    """``memory_write``: a value is about to be written to memory. ``payload`` = value."""

    memory_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryReadContext(InterceptionContext):
    """``memory_read``: a memory query is being issued. ``payload`` = query (pre) / results (post)."""

    memory_type: str | None = None
    query: str | None = None


@dataclass
class KnowledgeRetrievalContext(InterceptionContext):
    """``knowledge_retrieval``: a knowledge query. ``payload`` = query / retrieved results."""

    query: Any = None


@dataclass
class PreCodeExecutionContext(InterceptionContext):
    """``pre_code_execution``: code is about to run. ``payload`` = the code string."""

    code: str | None = None
    language: str | None = None


@dataclass
class MCPConnectContext(InterceptionContext):
    """``mcp_connect``: an MCP client is about to connect. ``payload`` = connection params."""

    server_name: str | None = None
    server_params: Any = None


@dataclass
class FileAccessContext(InterceptionContext):
    """``file_access``: reserved. No live consumer seam yet."""

    path: str | None = None
    mode: str | None = None


@dataclass
class ArtifactOutputContext(InterceptionContext):
    """``artifact_output``: reserved. No live consumer seam yet."""

    artifact: Any = None


@dataclass
class FlowTransitionContext(InterceptionContext):
    """``flow_transition``: a flow is moving to triggered methods."""

    from_method: str | None = None
    to_methods: list[str] = field(default_factory=list)
    trigger: str | None = None


@dataclass
class RouterDecisionContext(InterceptionContext):
    """``router_decision``: a flow router is choosing a route. ``payload`` = route label."""

    router_name: str | None = None
    route: Any = None
