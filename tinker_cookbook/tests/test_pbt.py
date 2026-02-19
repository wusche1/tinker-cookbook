import json
import os
import random
import tempfile

import pytest

from tinker_cookbook.pbt.mutations import apply_mutation, generate_branch_hyperparams
from tinker_cookbook.pbt.train import read_metric, select_winner
from tinker_cookbook.pbt.types import BranchResult, HyperparamMutation


# --- Mutation tests ---


class TestApplyMutation:
    def test_allowed_values(self):
        mutation = HyperparamMutation(
            param_name="learning_rate",
            allowed_values=[1e-4, 1e-3, 1e-2],
        )
        rng = random.Random(42)
        value = apply_mutation(5e-4, mutation, rng)
        assert value in [1e-4, 1e-3, 1e-2]

    def test_additive_delta(self):
        mutation = HyperparamMutation(
            param_name="kl_penalty_coef",
            additive_delta=0.01,
        )
        rng = random.Random(42)
        value = apply_mutation(0.1, mutation, rng)
        assert value == pytest.approx(0.1 + 0.01) or value == pytest.approx(0.1 - 0.01)

    def test_multiplicative_factor(self):
        mutation = HyperparamMutation(
            param_name="learning_rate",
            multiplicative_factor=2.0,
        )
        rng = random.Random(42)
        value = apply_mutation(1e-4, mutation, rng)
        assert value == pytest.approx(1e-4 * 2.0) or value == pytest.approx(1e-4 / 2.0)

    def test_clamp_min(self):
        mutation = HyperparamMutation(
            param_name="learning_rate",
            additive_delta=1.0,
            min_value=0.01,
        )
        rng = random.Random(0)
        # Force subtract by trying seeds until we get the negative branch
        for seed in range(100):
            rng = random.Random(seed)
            value = apply_mutation(0.05, mutation, rng)
            if value == pytest.approx(0.01):
                break
        assert value >= 0.01

    def test_clamp_max(self):
        mutation = HyperparamMutation(
            param_name="learning_rate",
            additive_delta=1.0,
            max_value=0.5,
        )
        rng = random.Random(0)
        for seed in range(100):
            rng = random.Random(seed)
            value = apply_mutation(0.05, mutation, rng)
            if value == pytest.approx(0.5):
                break
        assert value <= 0.5

    def test_clamp_both(self):
        mutation = HyperparamMutation(
            param_name="temperature",
            multiplicative_factor=10.0,
            min_value=0.1,
            max_value=2.0,
        )
        rng = random.Random(42)
        for _ in range(50):
            value = apply_mutation(1.0, mutation, rng)
            assert 0.1 <= value <= 2.0

    def test_no_strategy_raises(self):
        mutation = HyperparamMutation(param_name="lr")
        with pytest.raises(ValueError, match="No mutation strategy"):
            apply_mutation(1.0, mutation, random.Random(0))


class TestGenerateBranchHyperparams:
    def test_all_params_mutated(self):
        mutations = [
            HyperparamMutation(param_name="learning_rate", multiplicative_factor=2.0),
            HyperparamMutation(param_name="temperature", allowed_values=[0.5, 1.0, 1.5]),
        ]
        current = {"learning_rate": 1e-4, "temperature": 1.0}
        result = generate_branch_hyperparams(current, mutations, random.Random(42))
        assert "learning_rate" in result
        assert "temperature" in result
        assert len(result) == 2

    def test_respects_bounds(self):
        mutations = [
            HyperparamMutation(
                param_name="lr",
                multiplicative_factor=100.0,
                min_value=1e-6,
                max_value=1.0,
            ),
        ]
        current = {"lr": 0.001}
        for seed in range(50):
            result = generate_branch_hyperparams(current, mutations, random.Random(seed))
            assert 1e-6 <= result["lr"] <= 1.0


# --- Metric reading tests ---


class TestReadMetric:
    def test_reads_latest_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"step": 0, "reward": 0.1}) + "\n")
                f.write(json.dumps({"step": 1, "reward": 0.5}) + "\n")
                f.write(json.dumps({"step": 2, "reward": 0.9}) + "\n")
            assert read_metric(tmpdir, "reward") == pytest.approx(0.9)

    def test_missing_metric_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"step": 0, "other": 1.0}) + "\n")
            with pytest.raises(ValueError, match="not found"):
                read_metric(tmpdir, "reward")

    def test_metric_not_in_every_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"step": 0, "reward": 0.1}) + "\n")
                f.write(json.dumps({"step": 1, "other": 99.0}) + "\n")
            assert read_metric(tmpdir, "reward") == pytest.approx(0.1)


# --- Winner selection tests ---


class TestSelectWinner:
    def _make_result(self, branch_id: int, metric: float) -> BranchResult:
        return BranchResult(
            branch_id=branch_id,
            hyperparams={"lr": 0.001},
            metric_value=metric,
            state_path=f"/path/{branch_id}",
        )

    def test_maximize(self):
        results = [self._make_result(0, 0.5), self._make_result(1, 0.9), self._make_result(2, 0.3)]
        winner = select_winner(results, maximize=True)
        assert winner.branch_id == 1

    def test_minimize(self):
        results = [self._make_result(0, 0.5), self._make_result(1, 0.9), self._make_result(2, 0.3)]
        winner = select_winner(results, maximize=False)
        assert winner.branch_id == 2

    def test_single_result(self):
        results = [self._make_result(0, 0.5)]
        winner = select_winner(results, maximize=True)
        assert winner.branch_id == 0


# --- Config validation tests ---


class TestValidation:
    def _make_mutation(self, **kwargs) -> HyperparamMutation:
        defaults = {"param_name": "learning_rate", "multiplicative_factor": 2.0}
        defaults.update(kwargs)
        return HyperparamMutation(**defaults)

    def test_exactly_one_strategy_none(self):
        m = HyperparamMutation(param_name="lr")
        strategies = [
            m.allowed_values is not None,
            m.additive_delta is not None,
            m.multiplicative_factor is not None,
        ]
        assert sum(strategies) == 0

    def test_exactly_one_strategy_one(self):
        m = HyperparamMutation(param_name="lr", additive_delta=0.1)
        strategies = [
            m.allowed_values is not None,
            m.additive_delta is not None,
            m.multiplicative_factor is not None,
        ]
        assert sum(strategies) == 1

    def test_min_max_order(self):
        m = HyperparamMutation(
            param_name="lr", multiplicative_factor=2.0, min_value=0.1, max_value=1.0
        )
        assert m.min_value < m.max_value
