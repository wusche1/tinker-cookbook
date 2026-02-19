from dataclasses import dataclass

import chz

from tinker_cookbook.rl import train as rl_train


@chz.chz
class HyperparamMutation:
    param_name: str
    # Mutation strategy (exactly one must be set)
    allowed_values: list[float] | None = None
    additive_delta: float | None = None
    multiplicative_factor: float | None = None
    # Bounds
    min_value: float | None = None
    max_value: float | None = None


@chz.chz
class PBTConfig:
    rl_config: rl_train.Config
    mutations: list[HyperparamMutation]
    metric_name: str
    maximize_metric: bool = True
    pbt_every: int = 20
    explore_steps: int = 5
    population_size: int = 4
    include_baseline: bool = True
    start_with_exploration: bool = False


@dataclass
class BranchResult:
    branch_id: int | str
    hyperparams: dict[str, float]
    metric_value: float
    state_path: str


def validate_pbt_config(pbt_cfg: PBTConfig) -> None:
    cfg = pbt_cfg.rl_config

    if cfg.save_every <= 0:
        raise ValueError("save_every must be > 0 for PBT")
    if pbt_cfg.pbt_every % cfg.save_every != 0:
        raise ValueError(
            f"pbt_every ({pbt_cfg.pbt_every}) must be a multiple of save_every ({cfg.save_every})"
        )
    if pbt_cfg.population_size < 2:
        raise ValueError("population_size must be >= 2")
    if pbt_cfg.explore_steps < 1:
        raise ValueError("explore_steps must be >= 1")

    for mutation in pbt_cfg.mutations:
        strategies = [
            mutation.allowed_values is not None,
            mutation.additive_delta is not None,
            mutation.multiplicative_factor is not None,
        ]
        if sum(strategies) != 1:
            raise ValueError(
                f"Exactly one mutation strategy must be set for '{mutation.param_name}', "
                f"got {sum(strategies)}"
            )
        if not hasattr(cfg, mutation.param_name):
            raise ValueError(
                f"'{mutation.param_name}' is not a field on rl_train.Config"
            )
        if mutation.min_value is not None and mutation.max_value is not None:
            if mutation.min_value >= mutation.max_value:
                raise ValueError(
                    f"min_value ({mutation.min_value}) must be < max_value ({mutation.max_value}) "
                    f"for '{mutation.param_name}'"
                )

    if pbt_cfg.include_baseline and pbt_cfg.population_size < 2:
        raise ValueError(
            "population_size must be >= 2 when include_baseline=True "
            "(need at least 1 mutated branch + 1 baseline)"
        )


def make_branch_config(
    base_cfg: rl_train.Config,
    log_path: str,
    hyperparams: dict[str, float],
) -> rl_train.Config:
    return chz.replace(
        base_cfg,
        log_path=log_path,
        wandb_project=None,
        wandb_name=None,
        **hyperparams,
    )
