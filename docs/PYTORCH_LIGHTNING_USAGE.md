# PyTorch Lightning Usage in DSU

## Summary

**PyTorch Lightning is optional for inference.** We use a dummy module when it's not installed. There is **no performance difference** for inference with or without the real package.

## Where It's Used

| Location | Usage | Inference? |
|----------|-------|-------------|
| `models/bandit/core/` | `LightningModule` base class, `STEP_OUTPUT` type hint | Yes – Bandit models inherit from it |
| `models/bandit_v2/bandit.py` | `BaseEndToEndModule(pl.LightningModule)` | Yes – Bandit v2 models |
| `models/bandit/core/data/*/datamodule.py` | `LightningDataModule` | No – training only |
| `apollo/look2hear/models/base_model.py` | Lazy import in `serialize()` for `pl.__version__` | No – `serialize()` is for saving, not loading |
| `apollo/look2hear/utils/lightning_utils.py` | `rank_zero_only`, `rich_progress` | No – training UI |
| `apollo/look2hear/system/audio_litmodule.py` | `AudioLightningModule` | No – training |

## Inference Path

For inference we only use:

- `model.forward(batch)` – same as `nn.Module`
- `model.load_state_dict()` – from `nn.Module`
- `model.eval()` – from `nn.Module`

`LightningModule` adds training hooks (`training_step`, `validation_step`, `configure_optimizers`, etc.) that are **not used** during inference. Our `DummyLightningModule` is a plain `nn.Module`, so inference behavior is identical.

## Bandit Models in DSU

- `models.json` does **not** include any bandit models by default.
- Bandit is only used if someone adds a custom config with `model_type: bandit` or `bandit_v2`.
- The dummy ensures Bandit models still work when `pytorch-lightning` is not installed.

## Performance

- **No slowdown** – inference uses the same `forward()` path.
- **Smaller install** – skipping `pytorch-lightning` saves ~50MB and avoids its dependencies.

## Dummy Module

The dummy is injected in:

- `workers/audio_separator_worker.py` – before importing `audio_separator`
- `workers/bsroformer_worker.py` – before importing `utils.settings` (for Bandit via models.json)

It provides:

- `LightningModule` – minimal `nn.Module` subclass
- `utilities.rank_zero_only` – no-op decorator
- `utilities.types.STEP_OUTPUT` – `Optional[Any]`
- `callbacks.progress.rich_progress` – stub submodule
