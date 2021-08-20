import argparse
import glob
import logging
import os

import pandas as pd
import xarray as xr


"""
for V in hus1000 psl rlds rsds ta500 tas tos uas vas zg250 zg500; do
echo "SCRIPT: ${V}" >>process_monthlies.sh.out
python scratch/test.py /data/hpcdata/users/jambyr/icenet2/tom.data sh era5 $V >>process_monthlies.sh.out 2>>process_monthlies.sh.err
done


for I in nh sh; do
echo "SCRIPT: ${I}" >>process_monthlies_siconca.$I.out
python scratch/test.py /data/hpcdata/users/jambyr/icenet2/tom.data $I osisaf siconca >>process_monthlies_siconca.$I.out 2>>process_monthlies_siconca.$I.err
done

"""


def reprocess_monthlies(source, hemisphere, identifier, output_base,
                        dry=False,
                        var_names=[]):
    for var_name in var_names:
        var_path = os.path.join(source, hemisphere, var_name)
        files = glob.glob("{}/{}_*.nc".format(var_path, var_name))

        for file in files:
            _, year = os.path.basename(os.path.splitext(file)[0]).\
                          split("_")[0:2]

            try:
                year = int(year)

                if not (1900 < year < 2200):
                    logging.warning("File is not between 1900-2200, probably "
                                    "not something to convert: {}".format(year))
            except ValueError:
                logging.warning("Cannot derive year from {}".format(year))
                continue

            destination = os.path.join(output_base,
                                       identifier,
                                       hemisphere,
                                       var_name,
                                       str(year))

            if not os.path.exists(destination):
                os.makedirs(destination, exist_ok=True)

            logging.info("Processing {} from {} to {}".
                         format(var_name, year, destination))

            ds = xr.open_dataset(file)

            var_names = [name for name in list(ds.data_vars.keys())
                         if not name.startswith("lambert_")]

            var_names = set(var_names)
            logging.debug("Files have var names {} which will be renamed to {}".
                          format(", ".join(var_names), var_name))

            ds = ds.rename({k: var_name for k in var_names})
            da = getattr(ds, var_name)

            for date in da.time.values:
                date = pd.Timestamp(date)
                fname = '{:04d}_{:02d}_{:02d}.nc'. \
                    format(date.year, date.month, date.day)
                daily = da.sel(time=date)

                output_path = os.path.join(destination, fname)

                if dry or os.path.exists(output_path):
                    continue
                else:
                    daily.to_netcdf(output_path)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Temporary solution for reprocessing monthly files")

    a = argparse.ArgumentParser()
    a.add_argument("-d", "--dry", default=False, action="store_true")
    a.add_argument("-o", "--output", default="./data")
    a.add_argument("source")
    a.add_argument("hemisphere", choices=["nh", "sh"])
    a.add_argument("identifier")
    a.add_argument("vars", nargs='+')
    args = a.parse_args()

    reprocess_monthlies(args.source, args.hemisphere, args.identifier,
                        output_base=args.output,
                        dry=args.dry,
                        var_names=args.vars)

