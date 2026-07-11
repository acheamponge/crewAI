"""Conformance suite for the framework-native interception points.

For each wired point this suite asserts the shared contract: the probe hook
sees a well-shaped payload, an in-place/returned modification is honored, and a
:class:`HookAborted` interrupts the step. Enterprise / ACS adapters build
against these guarantees.
"""

from __future__ import annotations

from crewai.flow.flow import Flow, listen, router, start
from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear_all,
    on,
)
import pytest


@pytest.fixture(autouse=True)
def clear_dispatch_registry():
    clear_all()
    yield
    clear_all()


class _SimpleFlow(Flow):
    @start()
    def begin(self):
        return "begin"

    @listen(begin)
    def finish(self, _):
        return "flow-result"


class TestFlowExecutionBoundaries:
    """execution_start / input / output / execution_end on a flow."""

    def test_all_boundary_points_fire_once(self):
        fired: list[str] = []

        for point in (
            InterceptionPoint.EXECUTION_START,
            InterceptionPoint.INPUT,
            InterceptionPoint.OUTPUT,
            InterceptionPoint.EXECUTION_END,
        ):

            @on(point)
            def _probe(ctx, _point=point):
                fired.append(_point.value)

        _SimpleFlow().kickoff(inputs={"seed": 1})

        assert fired == [
            "execution_start",
            "input",
            "output",
            "execution_end",
        ]

    def test_output_modification_is_honored(self):
        @on(InterceptionPoint.OUTPUT)
        def rewrite(ctx):
            return "intercepted"

        result = _SimpleFlow().kickoff()
        assert result == "intercepted"

    def test_input_payload_carries_inputs(self):
        seen: dict = {}

        @on(InterceptionPoint.INPUT)
        def capture(ctx):
            seen.update(ctx.payload or {})

        _SimpleFlow().kickoff(inputs={"seed": 42})
        assert seen == {"seed": 42}

    def test_abort_at_execution_start_interrupts(self):
        @on(InterceptionPoint.EXECUTION_START)
        def block(ctx):
            raise HookAborted(reason="not allowed", source="policy")

        with pytest.raises(HookAborted) as exc:
            _SimpleFlow().kickoff()
        assert exc.value.reason == "not allowed"


class TestFlowStepPoints:
    """pre_step / post_step for flow methods (kind=flow_method)."""

    def test_pre_and_post_step_fire_per_method(self):
        kinds: list[tuple[str, str | None]] = []

        @on(InterceptionPoint.PRE_STEP)
        def pre(ctx):
            kinds.append(("pre", ctx.step_name))

        @on(InterceptionPoint.POST_STEP)
        def post(ctx):
            kinds.append(("post", ctx.step_name))

        _SimpleFlow().kickoff()

        assert ("pre", "begin") in kinds
        assert ("post", "begin") in kinds
        assert ("pre", "finish") in kinds
        assert ("post", "finish") in kinds

    def test_post_step_can_rewrite_method_output(self):
        @on(InterceptionPoint.POST_STEP)
        def rewrite(ctx):
            if ctx.step_name == "finish":
                return "rewritten"
            return None

        assert _SimpleFlow().kickoff() == "rewritten"


class _RouterFlow(Flow):
    @start()
    def begin(self):
        return "begin"

    @router(begin)
    def route(self):
        return "go_left"

    @listen("go_left")
    def left(self):
        return "left"

    @listen("go_right")
    def right(self):
        return "right"


class TestFlowTransitionAndRouter:
    """flow_transition and router_decision on a routed flow."""

    def test_transition_payload_carries_from_and_to(self):
        seen: list[tuple[str | None, list[str]]] = []

        @on(InterceptionPoint.FLOW_TRANSITION)
        def capture(ctx):
            seen.append((ctx.from_method, list(ctx.to_methods)))

        _RouterFlow().kickoff()

        assert any(to == ["left"] for _from, to in seen)

    def test_router_decision_fires_with_route(self):
        routes: list[object] = []

        @on(InterceptionPoint.ROUTER_DECISION)
        def capture(ctx):
            routes.append(ctx.route)

        _RouterFlow().kickoff()
        assert "go_left" in routes

    def test_router_decision_can_reroute(self):
        @on(InterceptionPoint.ROUTER_DECISION)
        def reroute(ctx):
            return "go_right"

        landed: list[str] = []

        @on(InterceptionPoint.PRE_STEP)
        def track(ctx):
            landed.append(ctx.step_name)

        _RouterFlow().kickoff()
        assert "right" in landed
        assert "left" not in landed
