from types import SimpleNamespace
from unittest.mock import Mock

from icenet.model.handlers import wandb as wandb_handler


def test_init_wandb_uses_metrics_logger(monkeypatch):
    """Use the non-deprecated Keras metrics callback."""
    callback = object()
    run = Mock()
    metrics_logger = Mock(return_value=callback)
    deprecated_callback = Mock()
    wandb = SimpleNamespace(
        init=Mock(return_value=run),
        Settings=Mock(),
        keras=SimpleNamespace(WandbCallback=deprecated_callback),
        integration=SimpleNamespace(
            keras=SimpleNamespace(WandbMetricsLogger=metrics_logger)
        ),
    )
    cli_args = SimpleNamespace(
        horovod=False,
        wandb_project="project",
        run_name="run",
        seed=1,
        preload=None,
        wandb_user="user",
        lr=0.001,
        filter_size=3,
        n_filters_factor=1,
        lr_10e_decay_fac=0.1,
        lr_decay_start=0,
        lr_decay_end=1,
        batch_size=2,
        wandb_offline=False,
    )
    monkeypatch.setattr(wandb_handler, "wandb", wandb, raising=False)
    monkeypatch.setattr(wandb_handler, "wandb_available", True)

    actual_run, actual_callback = wandb_handler.init_wandb(cli_args)

    assert actual_run is run
    assert actual_callback is callback
    metrics_logger.assert_called_once_with()
    deprecated_callback.assert_not_called()


def test_init_wandb_without_dependency(monkeypatch):
    """Leave W&B disabled when its optional dependency is unavailable."""
    monkeypatch.setattr(wandb_handler, "wandb_available", False)

    assert wandb_handler.init_wandb(SimpleNamespace()) == (None, None)
