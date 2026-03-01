"""Handler registry: maps type strings to handler instances."""
from __future__ import annotations

from typing import Any

from attractor.model.types import Node

SHAPE_TO_TYPE: dict[str, str] = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}


class HandlerRegistry:
    def __init__(self, default_handler: Any = None) -> None:
        self._handlers: dict[str, Any] = {}
        self._default_handler = default_handler

    def register(self, type_string: str, handler: Any) -> None:
        self._handlers[type_string] = handler

    def set_default(self, handler: Any) -> None:
        self._default_handler = handler

    def resolve(self, node: Node) -> Any:
        # 1. Explicit type attribute
        if node.type and node.type in self._handlers:
            return self._handlers[node.type]

        # 2. Shape-based resolution
        handler_type = SHAPE_TO_TYPE.get(node.shape)
        if handler_type and handler_type in self._handlers:
            return self._handlers[handler_type]

        # 3. Default
        return self._default_handler

    def build_default(
        self,
        backend: Any = None,
        interviewer: Any = None,
        runner: Any = None,
    ) -> "HandlerRegistry":
        """Populate the registry with all built-in handlers."""
        from attractor.handlers.start import StartHandler
        from attractor.handlers.exit_ import ExitHandler
        from attractor.handlers.conditional import ConditionalHandler
        from attractor.handlers.codergen import CodergenHandler
        from attractor.handlers.wait_human import WaitForHumanHandler
        from attractor.handlers.parallel import ParallelHandler
        from attractor.handlers.fan_in import FanInHandler
        from attractor.handlers.tool import ToolHandler
        from attractor.handlers.manager_loop import ManagerLoopHandler
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        if interviewer is None:
            interviewer = AutoApproveInterviewer()

        parallel_handler = ParallelHandler(runner=runner)

        self.register("start", StartHandler())
        self.register("exit", ExitHandler())
        self.register("conditional", ConditionalHandler())
        self.register("codergen", CodergenHandler(backend=backend))
        self.register("wait.human", WaitForHumanHandler(interviewer=interviewer))
        self.register("parallel", parallel_handler)
        self.register("parallel.fan_in", FanInHandler())
        self.register("tool", ToolHandler())
        self.register("stack.manager_loop", ManagerLoopHandler())
        self.set_default(CodergenHandler(backend=backend))

        return self


def create_default_registry(
    backend: Any = None,
    interviewer: Any = None,
    runner: Any = None,
) -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.build_default(backend=backend, interviewer=interviewer, runner=runner)
    return registry
