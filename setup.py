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
        "console_scripts": [
            # Data retrieval
            "gen_masks=icenet2.data.masks:generate",
            "download_sic_data=icenet2.data.sic:download",
            # TODO: reflect to appropriate module, not direct i/f to era5?
            "download_forecast_data=icenet2.data.climate:download",
            "regrid_forecast_data=icenet2.data.climate:regrid_data",
            "regrid_wind_data=icenet2.data.climate:regrid_wind_data",

            # Data loader / configuration
            #"preprocess=icenet2.data.loader:generate"

            # Model operations
        ],
    },
    python_requires='>=3.6, <4',
    install_requires=[

    ],
    include_package_data=True,
)
