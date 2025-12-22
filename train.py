import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb

from arc.dataset import Dataset
import arc.mdl as mdl
import arc.varc as varc


MODEL_REGISTRY = {"mdl": mdl, "varc": varc}


@dataclass
class Config:
    data_path: str = "data/arc"
    rearc_path: Optional[str] = "data/rearc"
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 10
    d_ff: int = 2048
    dropout: float = 0.1
    num_task_tokens: int = 1
    dtype: str = "bfloat16"
    seed: int = 0
    log_every: int = 10
    wandb_project: str = "arc-compare"
    wandb_run_name: Optional[str] = None
    max_grad_norm: float = 1.0
    model: str = "varc"


def make_train_step(optimizer: optax.GradientTransformation, loss_fn):
    def train_step(
        params,
        static,
        opt_state: optax.OptState,
        batch: Dict[str, jax.Array],
        key: jax.Array,
    ):
        def compute_loss(p):
            model = eqx.combine(p, static)
            return loss_fn(model, batch, key=key, inference=False)

        (loss, metrics), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            params
        )
        grads = jax.lax.pmean(grads, axis_name="devices")
        _ = jax.lax.pmean(loss, axis_name="devices")
        metrics = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, "devices"), metrics
        )

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return params, static, opt_state, metrics

    return train_step


def make_eval_step(loss_fn):
    def eval_step(
        params,
        static,
        batch: Dict[str, jax.Array],
        key: jax.Array,
    ):
        _, metrics = loss_fn(
            eqx.combine(params, static), batch, key=key, inference=True
        )
        return jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "devices"), metrics)

    return eval_step


def shard_batch(batch: Dict[str, jax.Array], num_devices: int) -> Dict[str, jax.Array]:
    def _reshape(x):
        return x.reshape(num_devices, x.shape[0] // num_devices, *x.shape[1:])

    return jax.tree_util.tree_map(_reshape, batch)


def create_datasets(config: Config):
    train_dataset = Dataset(
        path=Path(config.data_path),
        extra_train_path=Path(config.rearc_path) if config.rearc_path else None,
        split="training",
        subset="train",
        batch_size=config.batch_size,
        seed=config.seed,
    )

    eval_dataset = Dataset(
        path=Path(config.data_path),
        split="training",
        subset="test",
        task_lookup=train_dataset.task_lookup,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )

    return train_dataset, eval_dataset


def evaluate_model(
    params,
    static,
    eval_dataset,
    p_eval_step,
    eval_key: jax.Array,
    num_devices: int,
):
    metrics_sum = None

    for batch in eval_dataset:
        shard = shard_batch(batch, num_devices)

        eval_key, step_key = jax.random.split(eval_key)
        device_keys = jax.random.split(step_key, num_devices)

        metrics = p_eval_step(params, static, shard, device_keys)
        host_metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)

        metrics_sum = (
            jax.tree_util.tree_map(lambda a, b: a + b, metrics_sum, host_metrics)
            if metrics_sum is not None
            else host_metrics
        )

    return jax.tree_util.tree_map(lambda x: x / len(eval_dataset), metrics_sum)


def main(config: Config) -> None:
    devices = jax.local_devices()
    num_devices = len(devices)
    assert config.batch_size % num_devices == 0

    key = jax.random.PRNGKey(config.seed)
    model_key, train_key, eval_key = jax.random.split(key, 3)

    train_dataset, eval_dataset = create_datasets(config)

    model_module = MODEL_REGISTRY[config.model]
    model = model_module.build_model(config, num_tasks=train_dataset.num_tasks, key=model_key)
    loss_fn = model_module.loss_fn

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=min(config.epochs // 5, 10) * len(train_dataset),
        decay_steps=config.epochs * len(train_dataset),
        end_value=0.0,
    )

    params, static = eqx.partition(model, eqx.is_array)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay),
    )
    opt_state = optimizer.init(params)

    params = jax.device_put_replicated(params, devices)
    static = jax.device_put_replicated(static, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    train_step = make_train_step(optimizer, loss_fn)
    p_train_step = jax.pmap(train_step, axis_name="devices")

    eval_step = make_eval_step(loss_fn)
    p_eval_step = jax.pmap(eval_step, axis_name="devices")

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )

    wandb.log(
        {
            "data/num_train_tasks": train_dataset.num_tasks,
            "data/num_train_examples": train_dataset.num_samples,
            "data/train_batches_per_epoch": len(train_dataset),
            "data/train_total_steps": len(train_dataset) * config.epochs,
            "data/num_eval_tasks": eval_dataset.num_tasks,
            "data/num_eval_examples": eval_dataset.num_samples,
            "data/eval_batches_per_epoch": len(eval_dataset),
        }
    )

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        step_time_sum = 0.0
        step_time_count = 0

        for step, batch in enumerate(train_dataset):
            shard = shard_batch(batch, num_devices=num_devices)

            train_key, step_key = jax.random.split(train_key)
            device_keys = jax.random.split(step_key, num_devices)

            t0 = time.time()
            params, static, opt_state, metrics = p_train_step(
                params, static, opt_state, shard, device_keys
            )
            _ = list(metrics.values())[0].block_until_ready()
            step_time = time.time() - t0

            step_time_sum += step_time
            step_time_count += 1
            global_step += 1

            if step % config.log_every == 0:
                avg_step_time = step_time_sum / max(step_time_count, 1)
                host_metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)
                wandb.log(
                    {
                        **{f"train/{k}": v for k, v in host_metrics.items()},
                        "train/lr": float(lr_schedule(global_step)),
                        "train/time_per_batch_ms": avg_step_time * 1000.0,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
                step_time_sum = 0.0
                step_time_count = 0

        epoch_time = time.time() - epoch_start

        eval_key, epoch_key = jax.random.split(eval_key)
        eval_metrics = evaluate_model(
            params,
            static,
            eval_dataset,
            p_eval_step,
            epoch_key,
            num_devices,
        )

        os.makedirs("checkpoints", exist_ok=True)
        params_host = jax.tree_util.tree_map(lambda x: x[0], params)
        ckpt_path = os.path.join("checkpoints", f"{config.wandb_project}.eqx")
        eqx.tree_serialise_leaves(ckpt_path, eqx.combine(params_host, static))

        wandb.log(
            {
                "epoch": epoch,
                "epoch_time": epoch_time,
                "global_step": global_step,
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
            },
            step=global_step,
        )

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
