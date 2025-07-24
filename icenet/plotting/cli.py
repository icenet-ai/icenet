import argparse

from icenet.cli import date_arg
from icenet.utils import setup_logging


@setup_logging
def get_sample_get_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("date", type=date_arg)
    ap.add_argument("output_path", type=str, default="test.png")

    ap.add_argument("-c",
                    "--cols",
                    type=int,
                    default=8,
                    help="Plotting data over this number of columns")

    data_type = ap.add_mutually_exclusive_group(required=False)
    data_type.add_argument("--outputs", action="store_true", default=False)
    data_type.add_argument("--weights", action="store_true", default=False)

    ap.add_argument("-p", "--prediction", action="store_true", default=False)
    ap.add_argument("-s", "--size", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    args = ap.parse_args()
    return args


@setup_logging
def tfrecord_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("configuration", type=argparse.FileType("r"))
    ap.add_argument("-i", "--index", default=1, type=int)
    ap.add_argument("-l", "--levels", default=100, type=int)
    ap.add_argument("-o", "--output", default="plot")

    return ap.parse_args()


class ForecastPlotArgParser(argparse.ArgumentParser):
    """An ArgumentParser specialised to support forecast plot arguments

    Additional argument enabled by allow_ecmwf() etc.

    The 'allow_*' methods return self to permit method chaining.

    :param forecast_date: allows this positional argument to be disabled
    """

    def __init__(self, *args, forecast_date: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("obs_dataset_config", type=str)
        self.add_argument("forecast_file", type=str)
        if forecast_date:
            self.add_argument("forecast_date", type=date_arg)

        self.add_argument("-o", "--output-path", type=str, default=None)
        self.add_argument("-v",
                          "--verbose",
                          action="store_true",
                          default=False)
        self.add_argument("-r",
                          "--region",
                          default=None,
                          type=region_arg,
                          help="Region specified x1, y1, x2, y2")

    def allow_ecmwf(self):
        self.add_argument("-b",
                          "--bias-correct",
                          help="Bias correct SEAS forecast array",
                          action="store_true",
                          default=False)
        self.add_argument("-e", "--ecmwf", action="store_true", default=False)
        return self

    def allow_threshold(self):
        self.add_argument("-t",
                          "--threshold",
                          help="The SIC threshold of interest",
                          type=float,
                          default=0.15)
        return self

    def allow_sie(self):
        self.add_argument(
            "-ga",
            "--grid-area",
            help="The length of the sides of the grid used (in km)",
            type=int,
            default=25)
        return self

    def allow_metrics(self):
        self.add_argument("-m",
                          "--metrics",
                          help="Which metrics to compute and plot",
                          type=str,
                          default="mae,mse,rmse")
        self.add_argument(
            "-s",
            "--separate",
            help="Whether or not to produce separate plots for each metric",
            action="store_true",
            default=False)
        return self

    def allow_probes(self):
        self.add_argument(
            "-p",
            "--probe",
            action="append",
            dest="probes",
            type=location_arg,
            metavar="LOCATION",
            help="Sample at LOCATION",
        )
        return self

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            force=True,
        )
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        return args


def parse_location_or_region(argument: str):
    separator = ','
    # Allow ValueError to propagate if not given sequence of integers
    return tuple(int(s) for s in argument.split(separator))


def location_arg(argument: str):
    try:
        x, y = parse_location_or_region(argument)
        return x, y
    except ValueError:
        argparse.ArgumentTypeError(
            "Expected a location (pair of integers separated by a comma)")


def region_arg(argument: str):
    """type handler for region arguments with argparse

    :param argument:

    :return:
    """
    try:
        x1, y1, x2, y2 = parse_location_or_region(argument)

        if x2 < x1 or y2 < y1:
            raise RuntimeError(f"Region is not valid x1 {x1}:x2 {x2}, y1 {y1}:y2 {y2}")
        return x1, y1, x2, y2
    except TypeError:
        raise argparse.ArgumentTypeError(
            "Region argument must be list of four integers")


def parse_metrics_arg(argument: str) -> object:
    """
    Splits a string into a list by separating on commas.
    Will remove any whitespace and removes duplicates.
    Used to parsing metrics argument in metric_plots.

    :param argument: string

    :return: list of metrics to compute
    """
    return list(set([s.replace(" ", "") for s in argument.split(",")]))
