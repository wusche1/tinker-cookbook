"""
Integration test for PBT. Requires TINKER_API_KEY and network access.

Run with: uv run pytest tinker_cookbook/tests/test_pbt_integration.py -v -s
"""

import asyncio
import json
import os
import tempfile

import pytest

from tinker_cookbook.pbt.train import pbt_main
from tinker_cookbook.pbt.types import HyperparamMutation, PBTConfig
from tinker_cookbook.recipes.math_rl.arithmetic_env import ArithmeticDatasetBuilder
from tinker_cookbook.rl.train import Config


@pytest.fixture
def pbt_config():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tmpdir = os.path.join(repo_root, "tmp", "pbt_integration_test")
    os.makedirs(tmpdir, exist_ok=True)

    rl_config = Config(
        learning_rate=5e-4,
        dataset_builder=ArithmeticDatasetBuilder(
            batch_size=16,
            model_name_for_tokenizer=model_name,
            renderer_name="role_colon",
            n_batches=12,
            include_fewshot=True,
            group_size=4,
        ),
        model_name=model_name,
        lora_rank=16,
        max_tokens=5,
        log_path=tmpdir,
        save_every=4,
        eval_every=0,
        num_groups_to_log=0,
    )

    return PBTConfig(
        rl_config=rl_config,
        mutations=[
            HyperparamMutation(
                param_name="learning_rate",
                multiplicative_factor=2.0,
                min_value=1e-6,
                max_value=1e-2,
            ),
        ],
        metric_name="env/all/reward/total",
        maximize_metric=True,
        pbt_every=4,
        explore_steps=2,
        population_size=2,
        include_baseline=True,
        start_with_exploration=False,
    )


@pytest.mark.skipif(
    not os.environ.get("TINKER_API_KEY"),
    reason="TINKER_API_KEY not set",
)
def test_pbt_end_to_end(pbt_config):
    """Full PBT run: 12 batches, PBT every 4, explore 2 steps, population 2."""
    asyncio.run(pbt_main(pbt_config))

    log_path = pbt_config.rl_config.log_path

    # Trunk metrics should exist
    trunk_metrics_path = os.path.join(log_path, "metrics.jsonl")
    assert os.path.exists(trunk_metrics_path), "Trunk metrics.jsonl missing"

    # PBT state should exist
    pbt_state_path = os.path.join(log_path, "pbt_state.json")
    assert os.path.exists(pbt_state_path), "pbt_state.json missing"
    with open(pbt_state_path) as f:
        pbt_state = json.load(f)
    assert pbt_state["pbt_round"] >= 1, "Expected at least 1 PBT round"
    assert len(pbt_state["history"]) >= 1, "Expected history entries"
    assert "learning_rate" in pbt_state["current_hyperparams"]

    # Branch directories should exist for at least round 0
    round_dir = os.path.join(log_path, "pbt_round_000")
    assert os.path.isdir(round_dir), f"Missing PBT round directory: {round_dir}"

    # Each branch should have metrics
    branch_dirs = [d for d in os.listdir(round_dir) if os.path.isdir(os.path.join(round_dir, d))]
    assert len(branch_dirs) >= 2, f"Expected at least 2 branches, got {branch_dirs}"
    for branch_dir in branch_dirs:
        branch_metrics = os.path.join(round_dir, branch_dir, "metrics.jsonl")
        assert os.path.exists(branch_metrics), f"Missing metrics for {branch_dir}"

    # Final checkpoint should exist
    checkpoints_path = os.path.join(log_path, "checkpoints.jsonl")
    assert os.path.exists(checkpoints_path), "checkpoints.jsonl missing"

    print(f"\nPBT test passed! Logs at: {log_path}")
    print(f"PBT rounds completed: {pbt_state['pbt_round']}")
    print(f"Final hyperparams: {pbt_state['current_hyperparams']}")
    print(f"History: {json.dumps(pbt_state['history'], indent=2)}")
