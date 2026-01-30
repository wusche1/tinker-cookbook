"""
Integration test for seed-based reproducibility in RL training.

Runs 3 rollouts:
- Run 1: seed=42
- Run 2: seed=42 (should match Run 1)
- Run 3: seed=123 (should differ)
"""

import asyncio
from dataclasses import dataclass
from typing import Sequence
from unittest.mock import MagicMock, AsyncMock
import hashlib

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    StepResult,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.rl.rollouts import do_group_rollout


class SimpleEnv(Env):
    """A simple single-step environment for testing."""

    def __init__(self, problem_id: int):
        self.problem_id = problem_id
        self.step_count = 0

    async def initial_observation(self):
        # Return a simple observation
        ob = MagicMock(spec=tinker.ModelInput)
        ob.length = 10
        # Use problem_id to make observations distinguishable
        ob.__repr__ = lambda: f"Observation(problem={self.problem_id})"
        return ob, ["<end>"]

    async def step(self, action):
        self.step_count += 1
        # Simple reward based on action
        reward = sum(action) / (len(action) + 1)

        final_ob = MagicMock(spec=tinker.ModelInput)
        final_ob.length = 5

        return StepResult(
            reward=reward,
            episode_done=True,  # Single step episode
            next_observation=final_ob,
            next_stop_condition=["<end>"],
        )


@dataclass
class SimpleEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of simple test environments."""

    group_size: int = 4
    group_id: int = 0

    async def make_envs(self) -> Sequence[Env]:
        return [SimpleEnv(problem_id=self.group_id * self.group_size + i) for i in range(self.group_size)]

    def logging_tags(self):
        return ["test"]


def create_seed_dependent_mock_client(call_log: list):
    """
    Create a mock SamplingClient that returns deterministic results based on seed.
    Also logs all calls for comparison.
    """
    mock_client = MagicMock(spec=tinker.SamplingClient)

    async def mock_sample(prompt, num_samples, sampling_params):
        seed = sampling_params.seed

        # Generate deterministic tokens based on seed
        if seed is not None:
            # Use seed to generate deterministic "random" tokens
            hash_input = f"seed={seed}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()
            tokens = [int(b) % 100 for b in hash_bytes[:5]]
            logprobs = [-0.1 * (i + 1) for i in range(5)]
        else:
            # Without seed, use a "random" pattern (simulated)
            tokens = [1, 2, 3, 4, 5]
            logprobs = [-0.5] * 5

        # Log the call
        call_log.append({
            "seed": seed,
            "tokens": tokens,
        })

        # Create mock result
        mock_sequence = MagicMock()
        mock_sequence.tokens = tokens
        mock_sequence.logprobs = logprobs

        mock_result = MagicMock()
        mock_result.sequences = [mock_sequence]
        return mock_result

    mock_client.sample_async = AsyncMock(side_effect=mock_sample)
    return mock_client


async def run_rollout_with_seed(seed: int | None, group_id: int = 0) -> tuple[TrajectoryGroup, list]:
    """Run a single rollout with the given seed and return the trajectory group and call log."""
    call_log = []
    mock_client = create_seed_dependent_mock_client(call_log)

    policy = TinkerTokenCompleter(
        sampling_client=mock_client,
        max_tokens=100,
        temperature=1.0,
        seed=seed,
    )

    env_group_builder = SimpleEnvGroupBuilder(group_size=4, group_id=group_id)
    trajectory_group = await do_group_rollout(env_group_builder, policy)

    return trajectory_group, call_log


def extract_all_tokens(trajectory_group: TrajectoryGroup) -> list[list[int]]:
    """Extract all tokens from all trajectories in a group."""
    all_tokens = []
    for traj in trajectory_group.trajectories_G:
        traj_tokens = []
        for transition in traj.transitions:
            traj_tokens.extend(transition.ac.tokens)
        all_tokens.append(traj_tokens)
    return all_tokens


async def run_batch_with_per_rollout_seeds(base_seed: int | None, i_batch: int, batch_size: int) -> tuple[list[TrajectoryGroup], list[list]]:
    """
    Simulate how train.py computes per-rollout seeds.
    Each rollout in the batch gets: base_seed + i_batch * batch_size + i
    """
    all_traj_groups = []
    all_call_logs = []

    for i in range(batch_size):
        # This is the seed computation formula from train.py
        rollout_seed = base_seed + i_batch * batch_size + i if base_seed is not None else None

        call_log = []
        mock_client = create_seed_dependent_mock_client(call_log)

        policy = TinkerTokenCompleter(
            sampling_client=mock_client,
            max_tokens=100,
            temperature=1.0,
            seed=rollout_seed,
        )

        env_group_builder = SimpleEnvGroupBuilder(group_size=2, group_id=i)
        traj_group = await do_group_rollout(env_group_builder, policy)

        all_traj_groups.append(traj_group)
        all_call_logs.append(call_log)

    return all_traj_groups, all_call_logs


def test_seed_reproducibility():
    """Test that same seed produces same results, different seed produces different results."""

    async def run_test():
        print("="*60)
        print("TEST 1: Basic seed passing")
        print("="*60)

        # Run 1: seed=42
        print("\nRun 1: seed=42")
        traj_group_1, calls_1 = await run_rollout_with_seed(seed=42)
        tokens_1 = extract_all_tokens(traj_group_1)
        rewards_1 = traj_group_1.get_total_rewards()
        print(f"  Tokens: {tokens_1}")
        print(f"  Rewards: {rewards_1}")
        print(f"  Seeds used: {[c['seed'] for c in calls_1]}")

        # Run 2: seed=42 (should match Run 1)
        print("\nRun 2: seed=42 (should match Run 1)")
        traj_group_2, calls_2 = await run_rollout_with_seed(seed=42)
        tokens_2 = extract_all_tokens(traj_group_2)
        rewards_2 = traj_group_2.get_total_rewards()
        print(f"  Tokens: {tokens_2}")
        print(f"  Rewards: {rewards_2}")
        print(f"  Seeds used: {[c['seed'] for c in calls_2]}")

        # Run 3: seed=123 (should differ)
        print("\nRun 3: seed=123 (should differ)")
        traj_group_3, calls_3 = await run_rollout_with_seed(seed=123)
        tokens_3 = extract_all_tokens(traj_group_3)
        rewards_3 = traj_group_3.get_total_rewards()
        print(f"  Tokens: {tokens_3}")
        print(f"  Rewards: {rewards_3}")
        print(f"  Seeds used: {[c['seed'] for c in calls_3]}")

        # Assertions for Test 1
        print("\nVERIFICATION:")
        assert tokens_1 == tokens_2, f"Run 1 and Run 2 should have same tokens!"
        assert rewards_1 == rewards_2, f"Run 1 and Run 2 should have same rewards!"
        print("✓ Run 1 and Run 2 are IDENTICAL (same seed=42)")

        assert tokens_1 != tokens_3, f"Run 1 and Run 3 should have different tokens!"
        print("✓ Run 1 and Run 3 are DIFFERENT (different seeds: 42 vs 123)")

        assert all(c['seed'] == 42 for c in calls_1), "All calls in Run 1 should use seed=42"
        assert all(c['seed'] == 42 for c in calls_2), "All calls in Run 2 should use seed=42"
        assert all(c['seed'] == 123 for c in calls_3), "All calls in Run 3 should use seed=123"
        print("✓ Seeds were correctly passed to SamplingParams")

        print("\n" + "="*60)
        print("TEST 2: Per-rollout seed computation (as in train.py)")
        print("="*60)

        # Simulate batch processing with per-rollout seeds
        batch_size = 3
        base_seed = 1000

        # Batch 0 with base_seed=1000
        print(f"\nBatch 0: base_seed={base_seed}, batch_size={batch_size}")
        traj_groups_b0_run1, calls_b0_run1 = await run_batch_with_per_rollout_seeds(base_seed, i_batch=0, batch_size=batch_size)
        seeds_b0_run1 = [calls[0]['seed'] for calls in calls_b0_run1]
        print(f"  Seeds used: {seeds_b0_run1}")
        print(f"  Expected:   {[base_seed + 0 * batch_size + i for i in range(batch_size)]}")

        # Same batch again (should be identical)
        print(f"\nBatch 0 (repeat): base_seed={base_seed}")
        traj_groups_b0_run2, calls_b0_run2 = await run_batch_with_per_rollout_seeds(base_seed, i_batch=0, batch_size=batch_size)
        seeds_b0_run2 = [calls[0]['seed'] for calls in calls_b0_run2]
        print(f"  Seeds used: {seeds_b0_run2}")

        # Batch 1 (should have different seeds)
        print(f"\nBatch 1: base_seed={base_seed}")
        traj_groups_b1, calls_b1 = await run_batch_with_per_rollout_seeds(base_seed, i_batch=1, batch_size=batch_size)
        seeds_b1 = [calls[0]['seed'] for calls in calls_b1]
        print(f"  Seeds used: {seeds_b1}")
        print(f"  Expected:   {[base_seed + 1 * batch_size + i for i in range(batch_size)]}")

        # Verify Test 2
        print("\nVERIFICATION:")

        # Check seeds are computed correctly
        expected_b0 = [base_seed + 0 * batch_size + i for i in range(batch_size)]
        expected_b1 = [base_seed + 1 * batch_size + i for i in range(batch_size)]
        assert seeds_b0_run1 == expected_b0, f"Batch 0 seeds incorrect: {seeds_b0_run1} != {expected_b0}"
        assert seeds_b1 == expected_b1, f"Batch 1 seeds incorrect: {seeds_b1} != {expected_b1}"
        print(f"✓ Per-rollout seeds computed correctly: batch 0 = {expected_b0}, batch 1 = {expected_b1}")

        # Check all seeds are unique across batches
        all_seeds = seeds_b0_run1 + seeds_b1
        assert len(all_seeds) == len(set(all_seeds)), "All seeds should be unique"
        print("✓ All seeds across batches are unique")

        # Check same batch produces same results
        tokens_b0_run1 = [extract_all_tokens(tg) for tg in traj_groups_b0_run1]
        tokens_b0_run2 = [extract_all_tokens(tg) for tg in traj_groups_b0_run2]
        assert tokens_b0_run1 == tokens_b0_run2, "Same batch should produce same results"
        print("✓ Repeated batch 0 produces identical results")

        # Check different batches produce different results
        tokens_b1 = [extract_all_tokens(tg) for tg in traj_groups_b1]
        assert tokens_b0_run1 != tokens_b1, "Different batches should produce different results"
        print("✓ Different batches produce different results")

        print("\n" + "="*60)
        print("✅ All integration tests passed!")
        print("="*60)

    asyncio.run(run_test())


if __name__ == "__main__":
    test_seed_reproducibility()
