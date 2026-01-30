"""
Tests for seed-based reproducibility in RL training.

Tests verify that:
1. SeedGenerator produces deterministic, unique seeds
2. Same base seed produces identical rollout results
3. Different base seeds produce different results
4. Async training warns when seed is set (since async is non-deterministic)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence
from unittest.mock import MagicMock, AsyncMock, patch
import hashlib

import pytest
import tinker

from tinker_cookbook.completers import SeedGenerator, TinkerTokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    StepResult,
    TrajectoryGroup,
)
from tinker_cookbook.rl.rollouts import do_group_rollout


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class SimpleEnv(Env):
    """A simple single-step environment for testing."""

    def __init__(self, problem_id: int):
        self.problem_id = problem_id

    async def initial_observation(self):
        ob = MagicMock(spec=tinker.ModelInput)
        ob.length = 10
        return ob, ["<end>"]

    async def step(self, action):
        reward = sum(action) / (len(action) + 1)
        final_ob = MagicMock(spec=tinker.ModelInput)
        final_ob.length = 5
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=final_ob,
            next_stop_condition=["<end>"],
        )


@dataclass
class SimpleEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of simple test environments."""

    group_size: int = 4
    group_id: int = 0

    async def make_envs(self) -> Sequence[Env]:
        return [
            SimpleEnv(problem_id=self.group_id * self.group_size + i)
            for i in range(self.group_size)
        ]

    def logging_tags(self):
        return ["test"]


def create_seed_dependent_mock_client(call_log: list):
    """Create a mock SamplingClient that returns deterministic results based on seed."""
    mock_client = MagicMock(spec=tinker.SamplingClient)

    async def mock_sample(prompt, num_samples, sampling_params):
        seed = sampling_params.seed

        if seed is not None:
            hash_input = f"seed={seed}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()
            tokens = [int(b) % 100 for b in hash_bytes[:5]]
            logprobs = [-0.1 * (i + 1) for i in range(5)]
        else:
            tokens = [1, 2, 3, 4, 5]
            logprobs = [-0.5] * 5

        call_log.append({"seed": seed, "tokens": tokens})

        mock_sequence = MagicMock()
        mock_sequence.tokens = tokens
        mock_sequence.logprobs = logprobs

        mock_result = MagicMock()
        mock_result.sequences = [mock_sequence]
        return mock_result

    mock_client.sample_async = AsyncMock(side_effect=mock_sample)
    return mock_client


async def run_rollout_with_seed_generator(
    base_seed: int | None, group_size: int = 4
) -> tuple[TrajectoryGroup, list]:
    """Run a single rollout with a SeedGenerator and return results + call log."""
    call_log = []
    mock_client = create_seed_dependent_mock_client(call_log)

    seed_generator = SeedGenerator(base_seed)
    policy = TinkerTokenCompleter(
        sampling_client=mock_client,
        max_tokens=100,
        temperature=1.0,
        seed_generator=seed_generator,
    )

    env_group_builder = SimpleEnvGroupBuilder(group_size=group_size, group_id=0)
    trajectory_group = await do_group_rollout(env_group_builder, policy)

    return trajectory_group, call_log


def extract_all_tokens(trajectory_group: TrajectoryGroup) -> list[list[int]]:
    """Extract all tokens from all trajectories in a group."""
    return [
        [token for transition in traj.transitions for token in transition.ac.tokens]
        for traj in trajectory_group.trajectories_G
    ]


# =============================================================================
# SeedGenerator Unit Tests
# =============================================================================


def test_seed_generator_deterministic():
    """Same base seed produces same sequence of seeds."""
    gen1 = SeedGenerator(42)
    gen2 = SeedGenerator(42)

    seeds1 = [gen1.next_seed() for _ in range(10)]
    seeds2 = [gen2.next_seed() for _ in range(10)]

    assert seeds1 == seeds2


def test_seed_generator_different_base_seeds():
    """Different base seeds produce different sequences."""
    gen1 = SeedGenerator(42)
    gen2 = SeedGenerator(123)

    seeds1 = [gen1.next_seed() for _ in range(10)]
    seeds2 = [gen2.next_seed() for _ in range(10)]

    assert seeds1 != seeds2


def test_seed_generator_none_returns_none():
    """SeedGenerator with None base seed always returns None."""
    gen = SeedGenerator(None)

    for _ in range(10):
        assert gen.next_seed() is None


def test_seed_generator_unique_seeds():
    """Seeds within a sequence are unique."""
    gen = SeedGenerator(999)
    seeds = [gen.next_seed() for _ in range(1000)]

    assert len(seeds) == len(set(seeds))


def test_seed_generator_range():
    """Seeds are within valid range [0, 2^31 - 1]."""
    gen = SeedGenerator(12345)

    for _ in range(100):
        seed = gen.next_seed()
        assert seed is not None
        assert 0 <= seed < 2**31


# =============================================================================
# Rollout Reproducibility Tests
# =============================================================================


def test_same_seed_same_results():
    """Same base seed produces identical rollout results."""

    async def run():
        traj1, calls1 = await run_rollout_with_seed_generator(base_seed=42)
        traj2, calls2 = await run_rollout_with_seed_generator(base_seed=42)

        tokens1 = extract_all_tokens(traj1)
        tokens2 = extract_all_tokens(traj2)
        seeds1 = [c["seed"] for c in calls1]
        seeds2 = [c["seed"] for c in calls2]

        assert seeds1 == seeds2, "Same base seed should produce same seed sequence"
        assert tokens1 == tokens2, "Same base seed should produce same tokens"
        assert traj1.get_total_rewards() == traj2.get_total_rewards()

    asyncio.run(run())


def test_different_seed_different_results():
    """Different base seeds produce different rollout results."""

    async def run():
        traj1, calls1 = await run_rollout_with_seed_generator(base_seed=42)
        traj2, calls2 = await run_rollout_with_seed_generator(base_seed=123)

        tokens1 = extract_all_tokens(traj1)
        tokens2 = extract_all_tokens(traj2)
        seeds1 = [c["seed"] for c in calls1]
        seeds2 = [c["seed"] for c in calls2]

        assert seeds1 != seeds2, "Different base seeds should produce different sequences"
        assert tokens1 != tokens2, "Different base seeds should produce different tokens"

    asyncio.run(run())


def test_each_env_gets_unique_seed():
    """Each environment in a group gets a unique seed."""

    async def run():
        _, calls = await run_rollout_with_seed_generator(base_seed=42, group_size=8)
        seeds = [c["seed"] for c in calls]

        assert len(seeds) == len(set(seeds)), f"All seeds should be unique: {seeds}"

    asyncio.run(run())


def test_none_seed_no_determinism():
    """None base seed results in None seeds (non-deterministic sampling)."""

    async def run():
        _, calls = await run_rollout_with_seed_generator(base_seed=None)
        seeds = [c["seed"] for c in calls]

        assert all(s is None for s in seeds), "All seeds should be None"

    asyncio.run(run())


def test_multiple_batches_reproducible():
    """Multiple batches with shared SeedGenerator produce reproducible results."""

    async def run_batches(base_seed: int | None) -> list[int | None]:
        all_seeds = []
        seed_generator = SeedGenerator(base_seed)

        for batch_idx in range(3):
            for group_idx in range(2):
                call_log = []
                mock_client = create_seed_dependent_mock_client(call_log)

                policy = TinkerTokenCompleter(
                    sampling_client=mock_client,
                    max_tokens=100,
                    temperature=1.0,
                    seed_generator=seed_generator,
                )

                builder = SimpleEnvGroupBuilder(group_size=2, group_id=batch_idx * 2 + group_idx)
                await do_group_rollout(builder, policy)
                all_seeds.extend(c["seed"] for c in call_log)

        return all_seeds

    async def run():
        seeds1 = await run_batches(base_seed=1000)
        seeds2 = await run_batches(base_seed=1000)

        assert seeds1 == seeds2, "Same base seed should produce same sequence across batches"
        assert len(seeds1) == len(set(seeds1)), "All seeds should be unique"

    asyncio.run(run())


# =============================================================================
# Async Training Warning Test
# =============================================================================


def test_async_training_warns_when_seed_set():
    """Async training logs a warning when seed is configured."""
    from tinker_cookbook.rl.train import do_async_training, Config, AsyncConfig
    from tinker_cookbook.rl.types import RLDatasetBuilder, RLDataset

    # Create minimal mock config with seed set
    @dataclass
    class MockRLDataset(RLDataset):
        def get_batch(self, index: int):
            return []

        def __len__(self):
            return 0

    class MockDatasetBuilder(RLDatasetBuilder):
        async def __call__(self):
            return MockRLDataset(), None

    # We can't easily run the full async training, but we can check the warning
    # is issued at the start by patching the logger
    with patch("tinker_cookbook.rl.train.logger") as mock_logger:
        # Create a config with seed and async_config set
        # We'll just check the warning logic by examining the function's behavior

        # The warning is issued early in do_async_training, so we need to
        # trigger the function but can let it fail after the warning check
        async def run():
            try:
                # This will fail because we don't have real clients,
                # but the warning should be issued first
                await do_async_training(
                    start_batch=0,
                    end_batch=1,
                    num_batches=1,
                    cfg=MagicMock(
                        seed=42,  # Seed is set
                        async_config=MagicMock(groups_per_batch=1),
                        log_path="/tmp",
                        save_every=0,
                        ttl_seconds=3600,
                    ),
                    training_client=MagicMock(),
                    service_client=MagicMock(),
                    evaluators=[],
                    dataset=MockRLDataset(),
                    ml_logger=MagicMock(),
                    tokenizer=MagicMock(),
                )
            except Exception:
                pass  # Expected to fail, we just want the warning

        asyncio.run(run())

        # Check that warning was called with the expected message
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "seed" in warning_msg.lower()
        assert "async" in warning_msg.lower() or "non-deterministic" in warning_msg.lower()


def test_async_training_no_warning_when_seed_none():
    """Async training does not warn when seed is None."""
    with patch("tinker_cookbook.rl.train.logger") as mock_logger:

        async def run():
            from tinker_cookbook.rl.train import do_async_training

            try:
                await do_async_training(
                    start_batch=0,
                    end_batch=1,
                    num_batches=1,
                    cfg=MagicMock(
                        seed=None,  # No seed
                        async_config=MagicMock(groups_per_batch=1),
                        log_path="/tmp",
                        save_every=0,
                        ttl_seconds=3600,
                    ),
                    training_client=MagicMock(),
                    service_client=MagicMock(),
                    evaluators=[],
                    dataset=MagicMock(get_batch=MagicMock(return_value=[])),
                    ml_logger=MagicMock(),
                    tokenizer=MagicMock(),
                )
            except Exception:
                pass

        asyncio.run(run())

        # Warning should NOT have been called
        mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
