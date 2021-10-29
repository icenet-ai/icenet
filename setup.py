import setuptools

from setuptools import setup

"""Setup module for Icenet2 - draft module
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="icenet2",
    version="0.0.1a1",
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
            "icenet_data_masks = icenet2.data.sic.mask:main",

            "icenet_data_cmip = icenet2.data.interfaces.esgf:main",
            "icenet_data_era5 = icenet2.data.interfaces.cds:main",
            "icenet_data_hres = icenet2.data.interfaces.mars:main",
            "icenet_data_sic = icenet2.data.interfaces.osisaf:main",

            "icenet_data_reproc_monthly = "
            "icenet2.data.interfaces.utils:reprocess_main",

            "icenet_process_cmip = icenet2.data.processors.cmip:main",
            "icenet_process_era5 = icenet2.data.processors.era5:main",
            "icenet_process_hres = icenet2.data.processors.hres:main",
            "icenet_process_sic = icenet2.data.processors.osi:main",

            "icenet_process_metadata = icenet2.data.processors.meta:main",

            "icenet_dataset_create = icenet2.data.loader:main",

            "icenet_train = icenet2.model.train:main",
            "icenet_predict = icenet2.model.predict:main",

            "icenet_plot_set = icenet2.plotting.data:plot_set"
        ],
    },
    python_requires='>=3.6, <4',
    install_requires=[

    ],
    include_package_data=True,

    # TODO: sub-requirements
    # plotting: matplotlib, imageio, tqdm
)
