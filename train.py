# train_jax.py
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dataclasses import dataclass
import numpy as np
import torch

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb
import tyro

from mdl.dataset import _maybe_apply_color_aug_in_loop, build_torch_data

from mdl.model import Model, ModelConfig, compute_loss


@dataclass
class TrainConfig:
    num_workers: int = 0
    do_validate: bool = True
    data_path: Path = Path("data/arc-1/challenges.json")
    epochs: int = 101
    batch_size: int = 32
    val_batch_size: int = 300
    enable_color_aug_train: bool = True
    max_color_augments_train: int = 100
    enable_color_aug_eval: bool = False
    max_color_augments_eval: int = 0
    color_aug_seed: int = 42
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    seed: int = 42
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    n_layers: int = 4
    log_every: int = 10
    wandb_project: str = "mdl-arc"
    name: Optional[str] = "baseline"


def set_seed(seed: int) -> jax.Array:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return jax.random.PRNGKey(seed)


def torch_batch_to_jax(batch: Dict[str, Any]) -> Dict[str, Any]:
    input_ids = jnp.asarray(batch["input_ids"].cpu().numpy(), dtype=jnp.int32)
    attention_mask = jnp.asarray(batch["attention_mask"].cpu().numpy(), dtype=jnp.bool_)
    example_ids = jnp.asarray(batch["example_ids"].cpu().numpy(), dtype=jnp.int32)
    positions_3d = jnp.asarray(batch["positions_3d"].cpu().numpy(), dtype=jnp.int32)

    return {
        "input_ids": jax.device_put(input_ids),
        "attention_mask": jax.device_put(attention_mask),
        "example_ids": jax.device_put(example_ids),
        "positions_3d": jax.device_put(positions_3d),
    }


def _path_attr_names(path) -> Tuple[str, ...]:
    names = []
    for k in path:
        if hasattr(k, "name"):
            names.append(str(k.name))
        elif hasattr(k, "key"):
            names.append(str(k.key))
        elif hasattr(k, "idx"):
            names.append(str(k.idx))
        else:
            names.append(str(k))
    return tuple(names)


def make_weight_decay_mask(model: eqx.Module) -> Any:
    params = eqx.filter(model, eqx.is_inexact_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)

    mask_flat = []
    for path, leaf in flat:
        if leaf is None:
            mask_flat.append(False)
            continue

        names = _path_attr_names(path)
        last = names[-1] if names else ""
        in_attention = ("attention" in names) or ("attn" in names)

        in_embedding = ("token_embedding" in names) or ("example_embedding" in names)
        is_bias = last == "bias"
        is_weight = last == "weight"

        decay = bool(
            is_weight
            and (leaf.ndim >= 2)
            and (not in_attention)
            and (not in_embedding)
            and (not is_bias)
        )
        mask_flat.append(decay)

    return jax.tree_util.tree_unflatten(treedef, mask_flat)


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def train_step(
        model: eqx.Module,
        opt_state: optax.OptState,
        batch: Dict[str, jax.Array],
        key: jax.Array,
    ):
        key, drop_key = jax.random.split(key)

        def loss_fn(m):
            out = compute_loss(
                m,
                batch,
                key=drop_key,
                inference=False,
            )
            return out.loss, out

        (loss, out), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)

        metrics = {
            "loss": loss,
            "input_loss": out.input_loss,
            "output_loss": out.output_loss,
            "num_output_tokens": out.num_output_tokens,
        }
        return model, opt_state, metrics, key

    return train_step


def make_val_step():
    @eqx.filter_jit
    def val_step(model: eqx.Module, batch: Dict[str, jax.Array]):
        out = compute_loss(model, batch, key=None, inference=True)
        return out.output_loss, out.num_output_tokens

    return val_step


def main(cfg: TrainConfig) -> None:
    print(f"JAX backend={jax.default_backend()} devices={jax.devices()}")
    if not any(d.platform == "gpu" for d in jax.devices()):
        print(
            "WARNING: JAX is not running on GPU; expect very slow training. "
            "Check your CUDA-enabled jax/jaxlib install."
        )
    key = set_seed(cfg.seed)

    train_dataset, train_loader, val_loader, data_path, color_augmentor = (
        build_torch_data(cfg)
    )

    model_cfg = ModelConfig(
        num_examples=train_dataset.num_examples,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        dtype=jnp.bfloat16,
    )
    key, model_key = jax.random.split(key)
    model = Model(model_cfg, key=model_key)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(cfg.epochs)
    warmup_steps = int(total_steps * 0.05)

    warmup_f = jnp.asarray(float(warmup_steps), dtype=jnp.float32)
    total_f = jnp.asarray(float(total_steps), dtype=jnp.float32)
    lr_base = jnp.asarray(float(cfg.lr), dtype=jnp.float32)

    def lr_schedule(step):
        step_f = jnp.asarray(step, dtype=jnp.float32)
        step_m1 = step_f - 1.0

        warm_region = (step_m1 < warmup_f) & (warmup_f > 0)
        warm_factor = step_m1 / jnp.maximum(warmup_f, 1.0)

        decay_progress = (step_m1 - warmup_f) / jnp.maximum(total_f - warmup_f, 1.0)
        decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))

        base_factor = jnp.where(warm_region, warm_factor, jnp.maximum(0.0, decay_factor))
        factor = jnp.where(step_f <= 0.0, 1.0, base_factor)
        return lr_base * factor

    wd_mask = make_weight_decay_mask(model)

    transforms = []
    if float(cfg.grad_clip) > 0:
        transforms.append(optax.clip_by_global_norm(float(cfg.grad_clip)))
    transforms.extend(
        [
            optax.scale_by_adam(),
            optax.masked(optax.add_decayed_weights(float(cfg.weight_decay)), wd_mask),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1.0),
        ]
    )
    optimizer = optax.chain(*transforms)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_step = make_train_step(optimizer)
    val_step = make_val_step()

    wandb.init(
        project=getattr(cfg, "wandb_project", "mdl-arc"),
        name=getattr(cfg, "name", None),
        config={
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()
        },
    )

    global_step = 0
    log_every = int(getattr(cfg, "log_every", 10))
    metric_sums = {
        "loss": jnp.array(0.0, dtype=jnp.float32),
        "input_loss": jnp.array(0.0, dtype=jnp.float32),
        "output_loss": jnp.array(0.0, dtype=jnp.float32),
        "num_output_tokens": jnp.array(0, dtype=jnp.int32),
    }
    train_step_time_sum = 0.0
    metric_steps = 0

    for epoch in range(int(cfg.epochs)):
        if color_augmentor is not None and color_augmentor.num_permutations > 0:
            color_augmentor.set_index(epoch)
            print(
                f"Epoch {epoch+1}/{cfg.epochs} | "
                f"Using color permutation {color_augmentor.current_index + 1}/{color_augmentor.num_permutations}"
            )
        else:
            print(f"Epoch {epoch+1}/{cfg.epochs}")

        for batch in train_loader:
            _maybe_apply_color_aug_in_loop(
                batch,
                color_augmentor=getattr(train_loader, "color_augmentor", None),
                color_aug_in_collate=bool(
                    getattr(train_loader, "color_aug_in_collate", False)
                ),
            )
            jbatch = torch_batch_to_jax(batch)

            t_step_start = time.perf_counter()
            model, opt_state, metrics, key = train_step(model, opt_state, jbatch, key)
            jax.block_until_ready(metrics["loss"])
            t_step_end = time.perf_counter()
            global_step += 1
            metric_sums = {
                k: metric_sums[k] + metrics[k] for k in metric_sums
            }
            metric_steps += 1
            train_step_time_sum += t_step_end - t_step_start

            if log_every > 0 and global_step % log_every == 0:
                sums = jax.device_get(metric_sums)
                denom = float(max(metric_steps, 1))
                lr_now = float(jax.device_get(lr_schedule(global_step)))

                wandb.log(
                    {
                        "train/loss": float(sums["loss"]) / denom,
                        "train/input_loss": float(sums["input_loss"]) / denom,
                        "train/output_loss": float(sums["output_loss"]) / denom,
                        "train/num_output_tokens": int(sums["num_output_tokens"]) / denom,
                        "time/train_step_ms": (train_step_time_sum / denom) * 1e3,
                        "lr": lr_now,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
                metric_sums = {
                    "loss": jnp.array(0.0, dtype=jnp.float32),
                    "input_loss": jnp.array(0.0, dtype=jnp.float32),
                    "output_loss": jnp.array(0.0, dtype=jnp.float32),
                    "num_output_tokens": jnp.array(0, dtype=jnp.int32),
                }
                train_step_time_sum = 0.0
                metric_steps = 0

        if val_loader is not None:
            total_loss_sum = 0.0
            total_tokens = 0

            for batch in val_loader:
                if not any(batch.get("has_output", [True])):
                    continue

                jbatch = torch_batch_to_jax(batch)
                out_loss, num_out = val_step(model, jbatch)
                out_loss = float(jax.device_get(out_loss))
                num_out = int(jax.device_get(num_out))
                total_loss_sum += out_loss * max(0, num_out)
                total_tokens += max(0, num_out)

            val_output_loss = (
                (total_loss_sum / total_tokens) if total_tokens > 0 else 0.0
            )
            print(
                f"Epoch {epoch+1} | val/output_loss={val_output_loss:.6f} (tokens={total_tokens})"
            )
            wandb.log(
                {
                    "val/output_loss": float(val_output_loss),
                    "val/output_tokens": int(total_tokens),
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
