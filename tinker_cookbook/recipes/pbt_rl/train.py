import asyncio
import logging
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.pbt.train import pbt_main
from tinker_cookbook.pbt.types import HyperparamMutation, PBTConfig
from tinker_cookbook.recipes.math_rl import arithmetic_env
from tinker_cookbook.rl.train import Config

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Training
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5
    n_batches: int = 100

    # PBT
    pbt_every: int = 20
    explore_steps: int = 5
    population_size: int = 4
    start_with_exploration: bool = False

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    save_every: int = 20

    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_slug = cli_config.model_name.replace("/", "-")
    run_name = (
        f"pbt-arithmetic-{model_slug}-{cli_config.learning_rate}lr"
        f"-pop{cli_config.population_size}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/pbt_rl/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    rl_config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            n_batches=cli_config.n_batches,
            include_fewshot=True,
            group_size=cli_config.group_size,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        log_path=log_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=run_name,
        save_every=cli_config.save_every,
        eval_every=cli_config.save_every,
        base_url=cli_config.base_url,
    )

    pbt_config = PBTConfig(
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
        pbt_every=cli_config.pbt_every,
        explore_steps=cli_config.explore_steps,
        population_size=cli_config.population_size,
        start_with_exploration=cli_config.start_with_exploration,
    )

    await pbt_main(pbt_config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
