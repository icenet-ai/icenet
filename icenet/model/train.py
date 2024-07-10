import json
import logging
import time

import tensorflow as tf

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    pass

from icenet.data.dataset import IceNetDataSet, MergedIceNetDataSet
from icenet.model.cli import TrainingArgParser
from icenet.model.networks.tensorflow import HorovodNetwork, TensorflowNetwork, unet_batchnorm

from tensorflow.keras.models import load_model

import icenet.model.losses as losses
import icenet.model.metrics as metrics


def evaluate_model(model_path: object,
                   dataset: object,
                   dataset_ratio: float = 1.0):
    """

    :param model_path:
    :param dataset:
    :param dataset_ratio:
    """
    logging.info("Running evaluation against test set")
    network = load_model(model_path, compile=False)

    _, val_ds, test_ds = dataset.get_split_datasets(ratio=dataset_ratio)
    eval_data = val_ds

    if dataset.counts["test"] > 0:
        eval_data = test_ds
        logging.info("Using test set for validation")
    else:
        logging.warning("Using validation data source for evaluation, rather "
                        "than test set")

    lead_times = list(range(1, dataset.n_forecast_days + 1))
    logging.info("Metric creation for lead time of {} days".format(
        len(lead_times)))
    # TODO: common across train_model and evaluate_model - list of instantiations
    metric_names = ["binacc", "mae", "rmse"]
    metrics_classes = [
        metrics.WeightedBinaryAccuracy,
        metrics.WeightedMAE,
        metrics.WeightedRMSE,
    ]
    metrics_list = [
        cls(leadtime_idx=lt - 1) for lt in lead_times
        for cls in metrics_classes
    ]

    network.compile(weighted_metrics=metrics_list)

    logging.info('Evaluating... ')
    tic = time.time()
    results = network.evaluate(
        eval_data,
        return_dict=True,
        verbose=2
    )

    results_path = "{}.results.json".format(model_path)
    with open(results_path, "w") as fh:
        json.dump(results, fh)

    logging.debug(results)
    logging.info("Done in {:.1f}s".format(time.time() - tic))

    return results, metric_names, lead_times


def get_datasets(args):
    # TODO: this should come from a factory in the future - not the only place
    #  that merged datasets are going to be available

    dataset_filenames = [
        el if str(el).split(".")[-1] == "json" else "dataset_config.{}.json".format(el)
        for el in [args.dataset, *args.additional]
    ]

    if len(args.additional) == 0:
        dataset = IceNetDataSet(dataset_filenames[0],
                                batch_size=args.batch_size,
                                shuffling=args.shuffle_train)
    else:
        dataset = MergedIceNetDataSet(dataset_filenames,
                                      batch_size=args.batch_size,
                                      shuffling=args.shuffle_train)
    return dataset


def horovod_main():
    args = TrainingArgParser().add_unet().add_horovod().add_wandb().parse_args()
    hvd.init()

    if args.device_type in ("XPU", "GPU"):
        logging.debug("Setting up {} devices".format(args.device_type))
        devices = tf.config.list_physical_devices(args.device_type)
        logging.info("{} count is {}".format(args.device_type, len(devices)))

        for dev in devices:
            tf.config.experimental.set_memory_growth(dev, True)

        if devices:
            tf.config.experimental.set_visible_devices(devices[hvd.local_rank()], args.device_type)

    dataset = get_datasets(args)
    network = HorovodNetwork(dataset,
                             args.run_name,
                             checkpoint_mode=args.checkpoint_mode,
                             checkpoint_monitor=args.checkpoint_monitor,
                             early_stopping_patience=args.early_stopping,
                             lr_decay=(
                                 args.lr_10e_decay_fac,
                                 args.lr_decay_start,
                                 args.lr_decay_end,
                             ),
                             pre_load_path=args.preload,
                             seed=args.seed,
                             verbose=args.verbose)
    network.add_callback(
        hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    )

    execute_tf_training(args, dataset, network,
                        save=hvd.rank() == 0,
                        evaluate=hvd.rank() == 0)


def tensorflow_main():
    args = TrainingArgParser().add_unet().add_tensorflow().add_wandb().parse_args()
    dataset = get_datasets(args)
    network = TensorflowNetwork(dataset,
                                args.run_name,
                                checkpoint_mode=args.checkpoint_mode,
                                checkpoint_monitor=args.checkpoint_monitor,
                                early_stopping_patience=args.early_stopping,
                                lr_decay=(
                                    args.lr_10e_decay_fac,
                                    args.lr_decay_start,
                                    args.lr_decay_end,
                                ),
                                pre_load_path=args.preload,
                                seed=args.seed,
                                strategy=args.strategy,
                                verbose=args.verbose)
    execute_tf_training(args, dataset, network)


def execute_tf_training(args, dataset, network,
                        save=True,
                        evaluate=True):
    # There is a better way of doing this by passing off to a dynamic factory
    # for other integrations, but for the moment I have no shame
    using_wandb = False
    run = None

    # TODO: move to overridden implementation - decorator?
    if not args.no_wandb:
        from icenet.model.handlers.wandb import init_wandb, finalise_wandb
        run, callback = init_wandb(args)

        if callback is not None:
            network.add_callback(callback)
            using_wandb = True

    input_shape = (*dataset.shape, dataset.num_channels)
    ratio = args.ratio if args.ratio else 1.0
    train_ds, val_ds, _ = dataset.get_split_datasets(ratio=ratio)

    network.train(
        args.epochs,
        unet_batchnorm,
        train_ds,
        model_creator_kwargs=dict(
            input_shape=input_shape,
            loss=losses.WeightedMSE(),
            metrics=[
                metrics.WeightedBinaryAccuracy(),
                metrics.WeightedMAE(),
                metrics.WeightedRMSE(),
                losses.WeightedMSE()
            ],
            learning_rate=args.lr,
            filter_size=args.filter_size,
            n_filters_factor=args.n_filters_factor,
            n_forecast_days=dataset.n_forecast_days,
        ),
        save=save,
        validation_dataset=val_ds
    )

    if evaluate:
        results, metric_names, leads = \
            evaluate_model(network.model_path,
                           dataset,
                           dataset_ratio=args.ratio)

        if using_wandb:
            finalise_wandb(run, results, metric_names, leads)

