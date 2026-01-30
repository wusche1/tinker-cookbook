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

from tinker_cookbook.completers import SeedGenerator, TinkerTokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    StepResult,
    TrajectoryGroup,
)
from tinker_cookbook.rl.rollouts import do_group_rollout


class SimpleEnv(Env):
    """A simple single-step environment for testing."""

    def __init__(self, problem_id: int):
        self.problem_id = problem_id
        self.step_count = 0

    async def initial_observation(self):
        ob = MagicMock(spec=tinker.ModelInput)
        ob.length = 10
        ob.__repr__ = lambda: f"Observation(problem={self.problem_id})"
        return ob, ["<end>"]

    async def step(self, action):
        self.step_count += 1
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

        if seed is not None:
            hash_input = f"seed={seed}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()
            tokens = [int(b) % 100 for b in hash_bytes[:5]]
            logprobs = [-0.1 * (i + 1) for i in range(5)]
        else:
            tokens = [1, 2, 3, 4, 5]
            logprobs = [-0.5] * 5

        call_log.append({
            "seed": seed,
            "tokens": tokens,
        })

        mock_sequence = MagicMock()
        mock_sequence.tokens = tokens
        mock_sequence.logprobs = logprobs

        mock_result = MagicMock()
        mock_result.sequences = [mock_sequence]
        return mock_result

    mock_client.sample_async = AsyncMock(side_effect=mock_sample)
    return mock_client


async def run_rollout_with_seed_generator(base_seed: int | None, group_id: int = 0, group_size: int = 4) -> tuple[TrajectoryGroup, list]:
    """Run a single rollout with a SeedGenerator."""
    call_log = []
    mock_client = create_seed_dependent_mock_client(call_log)

    seed_generator = SeedGenerator(base_seed)

    policy = TinkerTokenCompleter(
        sampling_client=mock_client,
        max_tokens=100,
        temperature=1.0,
        seed_generator=seed_generator,
    )

    env_group_builder = SimpleEnvGroupBuilder(group_size=group_size, group_id=group_id)
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


def test_seed_generator_basics():
    """Test SeedGenerator produces deterministic, unique seeds."""
    print("Testing SeedGenerator basics...")

    # Same base seed should produce same sequence
    gen1 = SeedGenerator(42)
    gen2 = SeedGenerator(42)
    seeds1 = [gen1.next_seed() for _ in range(5)]
    seeds2 = [gen2.next_seed() for _ in range(5)]
    assert seeds1 == seeds2, f"Same base seed should produce same sequence: {seeds1} vs {seeds2}"
    print(f"✓ Same base seed (42) produces same sequence: {seeds1}")

    # Different base seed should produce different sequence
    gen3 = SeedGenerator(123)
    seeds3 = [gen3.next_seed() for _ in range(5)]
    assert seeds1 != seeds3, f"Different base seeds should produce different sequences"
    print(f"✓ Different base seed (123) produces different sequence: {seeds3}")

    # None seed should produce None
    gen_none = SeedGenerator(None)
    assert gen_none.next_seed() is None
    print("✓ None base seed produces None")

    # Seeds within a sequence should be unique
    gen4 = SeedGenerator(999)
    seeds4 = [gen4.next_seed() for _ in range(100)]
    assert len(seeds4) == len(set(seeds4)), "Seeds should be unique"
    print("✓ Seeds within a sequence are unique")


def test_seed_reproducibility():
    """Test that same seed produces same results, different seed produces different results."""

    async def run_test():
        print("\n" + "="*60)
        print("Testing seed reproducibility with SeedGenerator")
        print("="*60)

        # Run 1: seed=42
        print("\nRun 1: base_seed=42")
        traj_group_1, calls_1 = await run_rollout_with_seed_generator(base_seed=42, group_size=4)
        tokens_1 = extract_all_tokens(traj_group_1)
        rewards_1 = traj_group_1.get_total_rewards()
        seeds_1 = [c['seed'] for c in calls_1]
        print(f"  Seeds used: {seeds_1}")
        print(f"  Tokens: {tokens_1}")
        print(f"  Rewards: {rewards_1}")

        # Run 2: seed=42 (should match Run 1)
        print("\nRun 2: base_seed=42 (should match Run 1)")
        traj_group_2, calls_2 = await run_rollout_with_seed_generator(base_seed=42, group_size=4)
        tokens_2 = extract_all_tokens(traj_group_2)
        rewards_2 = traj_group_2.get_total_rewards()
        seeds_2 = [c['seed'] for c in calls_2]
        print(f"  Seeds used: {seeds_2}")
        print(f"  Tokens: {tokens_2}")
        print(f"  Rewards: {rewards_2}")

        # Run 3: seed=123 (should differ)
        print("\nRun 3: base_seed=123 (should differ)")
        traj_group_3, calls_3 = await run_rollout_with_seed_generator(base_seed=123, group_size=4)
        tokens_3 = extract_all_tokens(traj_group_3)
        rewards_3 = traj_group_3.get_total_rewards()
        seeds_3 = [c['seed'] for c in calls_3]
        print(f"  Seeds used: {seeds_3}")
        print(f"  Tokens: {tokens_3}")
        print(f"  Rewards: {rewards_3}")

        # Assertions
        print("\n" + "="*60)
        print("VERIFICATION:")
        print("="*60)

        # Check Run 1 == Run 2
        assert seeds_1 == seeds_2, f"Same base seed should produce same seed sequence"
        assert tokens_1 == tokens_2, f"Run 1 and Run 2 should have same tokens!"
        assert rewards_1 == rewards_2, f"Run 1 and Run 2 should have same rewards!"
        print("✓ Run 1 and Run 2 are IDENTICAL (same base_seed=42)")

        # Check Run 1 != Run 3
        assert seeds_1 != seeds_3, f"Different base seeds should produce different seed sequences"
        assert tokens_1 != tokens_3, f"Run 1 and Run 3 should have different tokens!"
        print("✓ Run 1 and Run 3 are DIFFERENT (different base seeds: 42 vs 123)")

        # Check each env in group got a DIFFERENT seed (key improvement!)
        assert len(set(seeds_1)) == len(seeds_1), f"Each env should get a unique seed, got {seeds_1}"
        print(f"✓ Each environment in group got a UNIQUE seed: {seeds_1}")

    asyncio.run(run_test())


def test_multiple_batches():
    """Test that multiple batches with same SeedGenerator produce reproducible results."""

    async def run_test():
        print("\n" + "="*60)
        print("Testing multiple batches with shared SeedGenerator")
        print("="*60)

        # Simulate 2 batches with 2 groups each, using a shared SeedGenerator
        async def run_two_batches(base_seed: int | None):
            all_calls = []
            seed_generator = SeedGenerator(base_seed)

            for batch_idx in range(2):
                for group_idx in range(2):
                    call_log = []
                    mock_client = create_seed_dependent_mock_client(call_log)

                    policy = TinkerTokenCompleter(
                        sampling_client=mock_client,
                        max_tokens=100,
                        temperature=1.0,
                        seed_generator=seed_generator,
                    )

                    env_group_builder = SimpleEnvGroupBuilder(group_size=2, group_id=batch_idx * 2 + group_idx)
                    await do_group_rollout(env_group_builder, policy)
                    all_calls.extend(call_log)

            return [c['seed'] for c in all_calls]

        # Run twice with same seed
        seeds_run1 = await run_two_batches(base_seed=1000)
        seeds_run2 = await run_two_batches(base_seed=1000)

        print(f"\nRun 1 seeds: {seeds_run1}")
        print(f"Run 2 seeds: {seeds_run2}")

        assert seeds_run1 == seeds_run2, "Same base seed should produce same sequence across batches"
        print("✓ Same base seed produces identical seed sequence across multiple batches")

        assert len(set(seeds_run1)) == len(seeds_run1), f"All seeds should be unique: {seeds_run1}"
        print(f"✓ All {len(seeds_run1)} seeds are unique")

    asyncio.run(run_test())


if __name__ == "__main__":
    print("Running seed implementation tests...\n")

    test_seed_generator_basics()
    print()

    test_seed_reproducibility()
    print()

    test_multiple_batches()

    print("\n" + "="*60)
    print("✅ All integration tests passed!")
    print("="*60)
