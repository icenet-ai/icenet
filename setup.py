import setuptools

from setuptools import setup

"""Setup module for Icenet2 - draft module
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="icenet2",
    version="0.0.1a0",
    author="Tom Andersson/James Byrne",
    author_email="jambyr@bas.ac.uk",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.bas.ac.uk",
    packages=setuptools.find_packages(),
    keywords="",
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 3 - Alpha",
    ],
    entry_points={
        # TODO: refactor to single entry point using click
        "console_scripts": [
            # 1. Pipeline used for provisioning
            # 2. Data retrieval
            # "gen_masks=icenet2.data.masks:generate",

            # "download_sic_data=icenet2.data.sic:download",

            # TODO: reflect to appropriate module?
            # TODO: not required for dailies? commented out for the mo
            # "download_era5=icenet2.data.climate.era5:cli",
            # "regrid_era5=icenet2.data.climate.era5:regrid_data",
            # "regrid_era5_wind=icenet2.data.climate.era5:regrid_wind_data",
            # "rotate_era5_wind=icenet2.data.climate.era5:rotate_wind_data",

            # "download_cmip6=icenet2.data.climate.cmip6:download",
            # "regrid_cmip6=icenet2.data.climate.cmip6:regrid_data",
            # "regrid_cmip6_wind=icenet2.data.climate.cmip6:regrid_wind_data",
            # "rotate_cmip6_wind=icenet2.data.climate.cmip6:rotate_wind_data",

            # "download_seas5=icenet2.data.sic.seas5:download",
            # "regrid_seas5=icenet2.data.sic.seas5:regrid_data",
            # "bias_correct_seas5=icenet2.data.sic.seas5:bias_correct",

            # 3. Process data loader / configuration
            # - get_configuration can use module instantiated caching to
            #   generate the configuration used for the dataloader
            "configure_icenet=icenet2.data.loader:cli",
            # - the data loader, be it tfrecord or numpy based
            "preprocess_icenet=icenet2.data.loader:cli",

            # 4 and 5. Use the model
            # TODO: discussion around use of wandb, avoiding this dependency
            "train_icenet=icenet2.model.train:cli",
            "predict_icenet=icenet2.model.predict:cli",

            # 6. Miscellaneous tools
            # TODO: plotting/analysis tools will be included here
            # "analyse_uncertainty=icenet.results.data.analyse_uncertainty",
        ],
    },
    python_requires='>=3.6, <4',
    install_requires=[

    ],
    include_package_data=True,

    # TODO: sub-requirements
    # plotting: matplotlib, imageio, tqdm
)
