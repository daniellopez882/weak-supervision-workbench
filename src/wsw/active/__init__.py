"""Active weak supervision: selection strategies and the simulated-oracle loop."""

from wsw.active.loop import (
    DEFAULT_BUDGETS,
    CurvePoint,
    LoopConfig,
    LoopResult,
    SimulatedOracle,
    Task,
    run_loop,
)
from wsw.active.strategies import STRATEGIES, Pool, Strategy, make_strategy, posterior_margin

__all__ = [
    "DEFAULT_BUDGETS",
    "STRATEGIES",
    "CurvePoint",
    "LoopConfig",
    "LoopResult",
    "Pool",
    "SimulatedOracle",
    "Strategy",
    "Task",
    "make_strategy",
    "posterior_margin",
    "run_loop",
]
