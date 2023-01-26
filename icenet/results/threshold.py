import argparse
import logging

import numpy as np
import xarray as xr

from icenet.data.cli import date_arg
from icenet.utils import setup_logging


def threshold_exceeds(da: object, sic_thresh: float, window_length: int = 1):
    """

    :param da:
    :param sic_thresh:
    :param window_length:
    :return:
    """
    logging.info("Checking thresholds for forecast(s)")

    thresh_arr = da > sic_thresh
    window_acc = thresh_arr.rolling(leadtime=window_length).reduce(np.sum)
    threshold_exceed_arr = xr.where(window_acc == window_length, da, 0)

    return np.argwhere(threshold_exceed_arr.values)


@setup_logging
def threshold_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--date-start", type=date_arg, default=None)
    ap.add_argument("-e", "--date-end", type=date_arg, default=None)

    ap.add_argument("-o", "--output-file", default=None, type=str)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    ap.add_argument("forecast_file")
    ap.add_argument("threshold", type=float)
    ap.add_argument("window_length", type=int)
    args = ap.parse_args()
    return args


def threshold_main():
    args = threshold_args()
    da = xr.open_dataset(args.forecast_file).sic_mean
    threshold_map = threshold_exceeds(da, args.threshold, args.window_length)

    # Currently naive implementation
    if args.output_file:
        with open(args.output_file, "wb") as fh:
            np.save(fh, threshold_map)
        logging.info("Saved to {}".format(args.output_file))
    else:
        logging.info("No output file provided: {} cells breached threshold".
                     format(len(threshold_map)))
