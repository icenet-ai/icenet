import datetime as dt
import logging

wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    pass


def init_wandb(cli_args):
    if wandb_available:
        logging.warning("Initialising WANDB for this run at user request")

        run = wandb.init(
            project=cli_args.wandb_project,
            name="{}.{}".format(cli_args.run_name, cli_args.seed),
            notes="{}: run at {}{}".format(
                cli_args.run_name,
                dt.datetime.now().strftime("%D %T"), "" if cli_args.preload is None
                else " preload {}".format(cli_args.preload)),
            entity=cli_args.wandb_user,
            config=dict(
                seed=cli_args.seed,
                learning_rate=cli_args.lr,
                filter_size=cli_args.filter_size,
                n_filters_factor=cli_args.n_filters_factor,
                lr_10e_decay_fac=cli_args.lr_10e_decay_fac,
                lr_decay_start=cli_args.lr_decay_start,
                lr_decay_end=cli_args.lr_decay_end,
                batch_size=cli_args.batch_size,
            ),
            settings=wandb.Settings(
                #    start_method="fork",
                #    _disable_stats=True,
            ),
            allow_val_change=True,
            mode='offline' if cli_args.wandb_offline else 'online',
            group=cli_args.run_name,
        )

        # Log training metrics to wandb each epoch
        return run, wandb.keras.WandbCallback(
            monitor=cli_args.checkpoint_monitor,
            mode=cli_args.checkpoint_mode,
            save_model=False,
            save_graph=False,
        )

    logging.warning("WandB is not available, we will never use it")
    return None, None


def finalise_wandb(run, results, metric_names, leads):
    logging.info("Updating wandb run with evaluation metrics")
    metric_vals = [[results[f'{name}{lt}'] for lt in leads]
                   for name in metric_names]
    table_data = list(zip(leads, *metric_vals))
    table = wandb.Table(data=table_data,
                        columns=['leadtime', *metric_names])

    # Log each metric vs. leadtime as a plot to wandb
    for name in metric_names:
        logging.debug("WandB logging {}".format(name))
        run.log(
            {f'{name}_plot': wandb.plot.line(table, x='leadtime', y=name)})
