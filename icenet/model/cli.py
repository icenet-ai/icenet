import argparse
import os

from icenet.utils import setup_logging

@setup_logging
def get_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("run_name", type=str)
    ap.add_argument("seed", type=int)

    ap.add_argument("-b", "--batch-size", type=int, default=4)
    ap.add_argument("-ca",
                    "--checkpoint-mode",
                    default="min",
                    type=str)
    ap.add_argument("-cm",
                    "--checkpoint-monitor",
                    default="val_rmse",
                    type=str)
    ap.add_argument("-ds",
                    "--additional-dataset",
                    dest="additional",
                    nargs="*",
                    default=[])
    ap.add_argument("-e", "--epochs", type=int, default=4)
    ap.add_argument("-f", "--filter-size", type=int, default=3)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("-m",
                    "--multiprocessing",
                    action="store_true",
                    default=False)
    ap.add_argument("-n", "--n-filters-factor", type=float, default=1.)
    ap.add_argument("-p", "--preload", type=str)
    ap.add_argument("-pw",
                    "--pickup-weights",
                    action="store_true",
                    default=False)
    ap.add_argument("-qs", "--max-queue-size", default=10, type=int)
    ap.add_argument("-r", "--ratio", default=1.0, type=float)

    ap.add_argument("-s",
                    "--strategy",
                    default=None,
                    choices=("default", "mirrored", "central"))

    ap.add_argument("--shuffle-train",
                    default=False,
                    action="store_true",
                    help="Shuffle the training set")
    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    ap.add_argument("-w", "--workers", type=int, default=4)

    # WandB additional arguments
    ap.add_argument("-nw", "--no-wandb", default=False, action="store_true")
    ap.add_argument("-wo",
                    "--wandb-offline",
                    default=False,
                    action="store_true")
    ap.add_argument("-wp",
                    "--wandb-project",
                    default=os.environ.get("ICENET_ENVIRONMENT"),
                    type=str)
    ap.add_argument("-wu",
                    "--wandb-user",
                    default=os.environ.get("USER"),
                    type=str)

    # Learning rate arguments
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--lr_10e_decay_fac",
                    default=1.0,
                    type=float,
                    help="Factor by which LR is multiplied by every 10 epochs "
                    "using exponential decay. E.g. 1 -> no decay (default)"
                    ", 0.5 -> halve every 10 epochs.")
    ap.add_argument('--lr_decay_start', default=10, type=int)
    ap.add_argument('--lr_decay_end', default=30, type=int)

    return ap.parse_args()