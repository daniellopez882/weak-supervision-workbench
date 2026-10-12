from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given
from hypothesis import strategies as st

from wsw.active import (
    STRATEGIES,
    LoopConfig,
    Pool,
    SimulatedOracle,
    Task,
    make_strategy,
    posterior_margin,
    run_loop,
)
from wsw.active.strategies import top_n
from wsw.data import load_dataset
from wsw.endmodel import TextFeatures
from wsw.lf import ABSTAIN

A = ABSTAIN


def test_posterior_margin_hand_values() -> None:
    P = np.array([[0.5, 0.3, 0.2], [0.34, 0.33, 0.33], [1.0, 0.0, 0.0], [0.45, 0.45, 0.1]])
    np.testing.assert_allclose(posterior_margin(P), [0.2, 0.01, 1.0, 0.0], atol=1e-12)


def test_top_n_orders_by_score_and_breaks_ties_randomly() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.0])
    candidates = np.array([0, 1, 2, 3, 4])
    firsts = {int(top_n(scores, candidates, 1, np.random.default_rng(s))[0]) for s in range(30)}
    assert firsts == {1, 3}
    picked = top_n(scores, candidates, 3, np.random.default_rng(0))
    assert set(picked.tolist()) == {1, 3, 2}
    assert top_n(scores, candidates[:0], 2, np.random.default_rng(0)).size == 0


def test_conflict_entropy_strategy_prefers_split_votes() -> None:
    L = np.array([[0, 0, A], [0, 1, A], [A, A, A], [1, 1, 1], [0, 1, 1]])
    pool = Pool(L, 2)
    labelled = np.zeros(5, dtype=bool)
    picked = make_strategy("conflict_entropy").select(
        pool, np.full((5, 2), 0.5), labelled, 1, np.random.default_rng(0)
    )
    assert picked.tolist() == [1]  # 1:1 split beats 1:2
    labelled[1] = True
    picked = make_strategy("conflict_entropy").select(
        pool, np.full((5, 2), 0.5), labelled, 1, np.random.default_rng(0)
    )
    assert picked.tolist() == [4]


def test_margin_strategy_picks_the_least_certain_row() -> None:
    P = np.array([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3]])
    picked = make_strategy("posterior_margin").select(
        Pool(np.zeros((3, 1), dtype=int), 2), P, np.zeros(3, bool), 2, np.random.default_rng(0)
    )
    assert picked.tolist() == [1, 2]


def test_coreset_picks_the_farthest_point() -> None:
    # Points on a line; with x=0 labelled the farthest is x=10, then the middle x=5.
    X = sp.csr_matrix(np.array([[0.0], [1.0], [5.0], [10.0], [9.0]]))
    labelled = np.array([True, False, False, False, False])
    strategy = make_strategy("coreset")
    pool = Pool(np.zeros((5, 1), dtype=int), 2, X)
    picked = strategy.select(pool, np.full((5, 2), 0.5), labelled, 2, np.random.default_rng(0))
    assert picked.tolist() == [3, 2]
    with pytest.raises(ValueError, match="feature"):
        make_strategy("coreset").select(
            Pool(np.zeros((5, 1), dtype=int), 2),
            np.full((5, 2), 0.5),
            labelled,
            1,
            np.random.default_rng(0),
        )


@given(
    st.sampled_from(sorted(STRATEGIES)),
    st.integers(5, 40),
    st.integers(0, 30),
    st.integers(1, 10),
    st.integers(0, 1000),
)
def test_selection_returns_distinct_unlabelled_rows(
    name: str, n: int, n_labelled: int, want: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    L = rng.integers(-1, 3, size=(n, 4))
    X = sp.csr_matrix(rng.random((n, 3)))
    P = rng.dirichlet(np.ones(3), size=n)
    labelled = np.zeros(n, dtype=bool)
    labelled[rng.choice(n, size=min(n_labelled, n), replace=False)] = True
    picked = make_strategy(name).select(Pool(L, 3, X), P, labelled, want, rng)
    assert picked.size == min(want, int((~labelled).sum()))
    assert len(set(picked.tolist())) == picked.size
    assert not labelled[picked].any()


def test_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        make_strategy("magic")


@pytest.fixture(scope="module")
def demo_task() -> tuple[Task, np.ndarray]:
    ds = load_dataset("demo")
    feats = TextFeatures.build(ds, min_df=1)
    task = Task(
        k=ds.k,
        pool_L=ds.train.L[:500],
        pool_X=feats.train[:500],
        test_L=ds.test.L,
        test_X=feats.test,
        test_y=ds.test.y,
        class_balance="uniform",
    )
    return task, ds.train.y[:500]


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_loop_is_deterministic_per_seed(strategy: str, demo_task: tuple[Task, np.ndarray]) -> None:
    task, y = demo_task
    config = LoopConfig(strategy=strategy, seed=3, budgets=(10, 30), batch_size=10)
    a = run_loop(task, SimulatedOracle(y), config)
    b = run_loop(task, SimulatedOracle(y), config)
    np.testing.assert_array_equal(a.selected, b.selected)
    assert a.points == b.points
    assert [p.n_labels for p in a.points] == [0, 10, 30]
    assert a.queries == 30
    assert len(set(a.selected.tolist())) == 30


def test_loop_seeds_differ_and_gold_only_skips_zero(demo_task: tuple[Task, np.ndarray]) -> None:
    task, y = demo_task
    a = run_loop(task, SimulatedOracle(y), LoopConfig("random", seed=0, budgets=(20,)))
    b = run_loop(task, SimulatedOracle(y), LoopConfig("random", seed=1, budgets=(20,)))
    assert a.selected.tolist() != b.selected.tolist()
    gold = run_loop(task, SimulatedOracle(y), LoopConfig("random", budgets=(20,), gold_only=True))
    assert [p.n_labels for p in gold.points] == [20]


def test_loop_with_unsupervised_label_model(demo_task: tuple[Task, np.ndarray]) -> None:
    task, y = demo_task
    result = run_loop(
        task, SimulatedOracle(y), LoopConfig("posterior_margin", budgets=(10,), label_model="metal")
    )
    assert result.points[-1].n_labels == 10


def test_loop_config_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        LoopConfig("random", budgets=(50, 25))
    with pytest.raises(ValueError, match="positive"):
        LoopConfig("random", budgets=(0, 25))
    with pytest.raises(ValueError, match="batch_size"):
        LoopConfig("random", batch_size=0)
