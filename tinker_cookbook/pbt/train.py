import asyncio
import json
import logging
import os
import random

import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.pbt.mutations import generate_branch_hyperparams
from tinker_cookbook.pbt.types import (
    BranchResult,
    PBTConfig,
    make_branch_config,
    validate_pbt_config,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.train import (
    Config,
    do_async_training,
    do_sync_training,
    do_sync_training_with_stream_minibatch,
)
from tinker_cookbook.rl.types import RLDataset
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.file_utils import read_jsonl
from tinker_cookbook.utils.trace import scope

logger = logging.getLogger(__name__)


def _get_training_func(cfg: Config):
    if cfg.async_config is not None:
        return do_async_training
    elif cfg.stream_minibatch_config is not None:
        return do_sync_training_with_stream_minibatch
    else:
        return do_sync_training


def _make_kl_reference_client(
    cfg: Config, service_client: tinker.ServiceClient
) -> tinker.SamplingClient | None:
    if cfg.kl_penalty_coef > 0:
        if cfg.kl_reference_config is None:
            raise ValueError("kl_reference_config must be specified when kl_penalty_coef > 0")
        return service_client.create_sampling_client(
            base_model=cfg.kl_reference_config.base_model,
            model_path=cfg.kl_reference_config.load_checkpoint_path,
        )
    return None


def _make_evaluators(
    cfg: Config, maybe_test_dataset: RLDataset | None
) -> list[SamplingClientEvaluator]:
    evaluators = [b() for b in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))
    return evaluators


def read_metric(log_path: str, metric_name: str) -> float:
    metrics_path = os.path.join(log_path, "metrics.jsonl")
    rows = read_jsonl(metrics_path)
    for row in reversed(rows):
        if metric_name in row:
            return float(row[metric_name])
    raise ValueError(f"Metric '{metric_name}' not found in {metrics_path}")


def select_winner(results: list[BranchResult], maximize: bool) -> BranchResult:
    if maximize:
        return max(results, key=lambda r: r.metric_value)
    else:
        return min(results, key=lambda r: r.metric_value)


def _log_branch_metrics_to_trunk(
    ml_logger: ml_log.Logger,
    log_path: str,
    pbt_round: int,
    results: list[BranchResult],
    step: int,
) -> None:
    round_dir = os.path.join(log_path, f"pbt_round_{pbt_round:03d}")
    merged: dict[str, float] = {}
    for result in results:
        branch_metrics_path = os.path.join(round_dir, f"branch_{result.branch_id}", "metrics.jsonl")
        if not os.path.exists(branch_metrics_path):
            continue
        rows = read_jsonl(branch_metrics_path)
        if not rows:
            continue
        last_row = rows[-1]
        last_row.pop("step", None)
        for k, v in last_row.items():
            merged[f"pbt_round_{pbt_round:03d}/branch_{result.branch_id}/{k}"] = v
    if merged:
        ml_logger.log_metrics(merged, step=step)


def _save_pbt_state(
    log_path: str,
    step: int,
    pbt_round: int,
    current_hyperparams: dict[str, float],
    state_path: str,
    history: list[dict],
) -> None:
    pbt_state = {
        "step": step,
        "pbt_round": pbt_round,
        "current_hyperparams": current_hyperparams,
        "state_path": state_path,
        "history": history,
    }
    path = os.path.join(log_path, "pbt_state.json")
    with open(path, "w") as f:
        json.dump(pbt_state, f, indent=2)
    logger.info(f"Saved PBT state to {path}")


def _load_pbt_state(log_path: str) -> dict | None:
    path = os.path.join(log_path, "pbt_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@scope
async def run_branch(
    branch_id: int | str,
    pbt_cfg: PBTConfig,
    state_path: str,
    start_batch: int,
    end_batch: int,
    num_batches: int,
    hyperparams: dict[str, float],
    dataset: RLDataset,
    maybe_test_dataset: RLDataset | None,
    service_client: tinker.ServiceClient,
    pbt_round: int,
) -> BranchResult:
    cfg = pbt_cfg.rl_config
    branch_log_path = os.path.join(
        cfg.log_path, f"pbt_round_{pbt_round:03d}", f"branch_{branch_id}"
    )
    branch_cfg = make_branch_config(cfg, branch_log_path, hyperparams)

    training_client = await service_client.create_training_client_from_state_with_optimizer_async(
        state_path
    )
    tokenizer = training_client.get_tokenizer()
    branch_logger = ml_log.setup_logging(
        log_dir=branch_log_path,
        config=branch_cfg,
        do_configure_logging_module=False,
    )
    evaluators = _make_evaluators(branch_cfg, maybe_test_dataset)
    kl_reference_client = _make_kl_reference_client(branch_cfg, service_client)
    training_func = _get_training_func(branch_cfg)

    logger.info(
        f"PBT round {pbt_round}, branch {branch_id}: "
        f"training steps [{start_batch}, {end_batch}), hyperparams={hyperparams}"
    )

    await training_func(
        start_batch=start_batch,
        end_batch=end_batch,
        num_batches=num_batches,
        cfg=branch_cfg,
        training_client=training_client,
        kl_reference_client=kl_reference_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=branch_logger,
        tokenizer=tokenizer,
    )

    paths = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="pbt_final",
        log_path=branch_log_path,
        loop_state={"batch": end_batch},
        kind="both",
        ttl_seconds=cfg.ttl_seconds,
    )

    metric_value = read_metric(branch_log_path, pbt_cfg.metric_name)
    branch_logger.close()

    logger.info(
        f"PBT round {pbt_round}, branch {branch_id}: {pbt_cfg.metric_name}={metric_value:.4f}"
    )

    return BranchResult(
        branch_id=branch_id,
        hyperparams=hyperparams,
        metric_value=metric_value,
        state_path=paths["state_path"],
    )


@scope
async def run_pbt_exploration(
    pbt_cfg: PBTConfig,
    state_path: str,
    current_hyperparams: dict[str, float],
    start_batch: int,
    end_batch: int,
    num_batches: int,
    dataset: RLDataset,
    maybe_test_dataset: RLDataset | None,
    service_client: tinker.ServiceClient,
    pbt_round: int,
) -> list[BranchResult]:
    rng = random.Random(pbt_round)

    num_mutated = pbt_cfg.population_size - (1 if pbt_cfg.include_baseline else 0)
    branch_hyperparams: list[tuple[int | str, dict[str, float]]] = []
    for i in range(num_mutated):
        hp = generate_branch_hyperparams(current_hyperparams, pbt_cfg.mutations, rng)
        branch_hyperparams.append((i, hp))
    if pbt_cfg.include_baseline:
        branch_hyperparams.append(("baseline", dict(current_hyperparams)))

    tasks = [
        run_branch(
            branch_id=bid,
            pbt_cfg=pbt_cfg,
            state_path=state_path,
            start_batch=start_batch,
            end_batch=end_batch,
            num_batches=num_batches,
            hyperparams=hp,
            dataset=dataset,
            maybe_test_dataset=maybe_test_dataset,
            service_client=service_client,
            pbt_round=pbt_round,
        )
        for bid, hp in branch_hyperparams
    ]

    return list(await asyncio.gather(*tasks))


@scope
async def pbt_main(pbt_cfg: PBTConfig) -> None:
    validate_pbt_config(pbt_cfg)
    cfg = pbt_cfg.rl_config

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=pbt_cfg,
        wandb_name=cfg.wandb_name,
    )

    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    # Resume from PBT state if available
    pbt_state = _load_pbt_state(cfg.log_path)
    if pbt_state is not None:
        step = pbt_state["step"]
        pbt_round = pbt_state["pbt_round"]
        current_hyperparams = pbt_state["current_hyperparams"]
        history = pbt_state["history"]
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                pbt_state["state_path"]
            )
        )
        logger.info(f"Resumed PBT from round {pbt_round}, step {step}")
    else:
        step = 0
        pbt_round = 0
        current_hyperparams = {m.param_name: getattr(cfg, m.param_name) for m in pbt_cfg.mutations}
        history: list[dict] = []

        # Create initial training client
        if cfg.load_checkpoint_path:
            training_client = await service_client.create_training_client_from_state_async(
                cfg.load_checkpoint_path
            )
            logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
        else:
            training_client = await service_client.create_lora_training_client_async(
                cfg.model_name, rank=cfg.lora_rank
            )

    tokenizer = training_client.get_tokenizer()
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    num_batches = len(dataset)
    logger.info(f"PBT training: {num_batches} total batches, pbt_every={pbt_cfg.pbt_every}")

    while step < num_batches:
        should_explore = (pbt_cfg.start_with_exploration and step == 0) or (
            step > 0 and step % pbt_cfg.pbt_every == 0
        )

        if should_explore:
            # Save trunk checkpoint for branching
            trunk_paths = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"pbt_trunk_{step:06d}",
                log_path=cfg.log_path,
                loop_state={"batch": step},
                kind="both",
                ttl_seconds=cfg.ttl_seconds,
            )

            explore_end = min(step + pbt_cfg.explore_steps, num_batches)
            results = await run_pbt_exploration(
                pbt_cfg=pbt_cfg,
                state_path=trunk_paths["state_path"],
                current_hyperparams=current_hyperparams,
                start_batch=step,
                end_batch=explore_end,
                num_batches=num_batches,
                dataset=dataset,
                maybe_test_dataset=maybe_test_dataset,
                service_client=service_client,
                pbt_round=pbt_round,
            )

            # Log branch metrics to trunk (appears in wandb under prefixed keys)
            _log_branch_metrics_to_trunk(
                ml_logger, cfg.log_path, pbt_round, results, step=explore_end
            )

            winner = select_winner(results, pbt_cfg.maximize_metric)
            logger.info(
                f"PBT round {pbt_round}: winner is branch {winner.branch_id} "
                f"with {pbt_cfg.metric_name}={winner.metric_value:.4f}, "
                f"hyperparams={winner.hyperparams}"
            )

            # Adopt winner
            training_client = (
                await service_client.create_training_client_from_state_with_optimizer_async(
                    winner.state_path
                )
            )
            current_hyperparams = winner.hyperparams
            tokenizer = training_client.get_tokenizer()

            # Log PBT decision
            pbt_metrics = {
                "pbt/round": pbt_round,
                "pbt/winner_branch": str(winner.branch_id),
                "pbt/metric_value": winner.metric_value,
            }
            for k, v in current_hyperparams.items():
                pbt_metrics[f"pbt/{k}"] = v
            ml_logger.log_metrics(pbt_metrics, step=explore_end)

            history.append(
                {
                    "round": pbt_round,
                    "step": step,
                    "winner_branch": str(winner.branch_id),
                    "hyperparams": current_hyperparams,
                    "metric": winner.metric_value,
                }
            )
            step = explore_end
            pbt_round += 1

            _save_pbt_state(
                cfg.log_path,
                step,
                pbt_round,
                current_hyperparams,
                winner.state_path,
                history,
            )
        else:
            # Normal training segment
            next_pbt = step + pbt_cfg.pbt_every - (step % pbt_cfg.pbt_every)
            train_end = min(next_pbt, num_batches)

            segment_cfg = make_branch_config(cfg, cfg.log_path, current_hyperparams)
            evaluators = _make_evaluators(segment_cfg, maybe_test_dataset)
            kl_reference_client = _make_kl_reference_client(segment_cfg, service_client)
            training_func = _get_training_func(segment_cfg)

            logger.info(
                f"PBT normal training: steps [{step}, {train_end}), "
                f"hyperparams={current_hyperparams}"
            )

            await training_func(
                start_batch=step,
                end_batch=train_end,
                num_batches=num_batches,
                cfg=segment_cfg,
                training_client=training_client,
                kl_reference_client=kl_reference_client,
                evaluators=evaluators,
                dataset=dataset,
                ml_logger=ml_logger,
                tokenizer=tokenizer,
            )

            step = train_end

    # Final checkpoint
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=cfg.log_path,
        loop_state={"batch": num_batches},
        kind="both",
        ttl_seconds=cfg.ttl_seconds,
    )

    _save_pbt_state(
        cfg.log_path,
        num_batches,
        pbt_round,
        current_hyperparams,
        "final",
        history,
    )

    ml_logger.close()
    logger.info("PBT training completed successfully")
