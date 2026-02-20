import random

from tinker_cookbook.pbt.types import HyperparamMutation


def apply_mutation(current_value: float, mutation: HyperparamMutation, rng: random.Random) -> float:
    if mutation.allowed_values is not None:
        value = rng.choice(mutation.allowed_values)
    elif mutation.additive_delta is not None:
        value = current_value + rng.choice([-mutation.additive_delta, mutation.additive_delta])
    elif mutation.multiplicative_factor is not None:
        value = current_value * rng.choice(
            [1.0 / mutation.multiplicative_factor, mutation.multiplicative_factor]
        )
    else:
        raise ValueError(f"No mutation strategy set for '{mutation.param_name}'")

    if mutation.min_value is not None:
        value = max(value, mutation.min_value)
    if mutation.max_value is not None:
        value = min(value, mutation.max_value)

    return value


def generate_branch_hyperparams(
    current_hyperparams: dict[str, float],
    mutations: list[HyperparamMutation],
    rng: random.Random,
) -> dict[str, float]:
    result = dict(current_hyperparams)
    for mutation in mutations:
        result[mutation.param_name] = apply_mutation(
            current_hyperparams[mutation.param_name], mutation, rng
        )
    return result
