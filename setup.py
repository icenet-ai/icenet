from setuptools import setup, find_packages

import icenet

"""Setup module for icenet
"""


def get_content(filename):
    with open(filename, "r") as fh:
        return fh.read()


setup(
    name=icenet.__name__,
    version=icenet.__version__,
    author=icenet.__author__,
    author_email=icenet.__email__,
    description="Library for operational IceNet forecasting",
    long_description="""{}\n---\n""".
                     format(get_content("README.md"),
                            get_content("HISTORY.rst")),
    long_description_content_type="text/markdown",
    url="https://github.com/icenet-ai",
    packages=find_packages(),
    keywords="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        ],
    entry_points={
        "console_scripts": [
            "icenet_data_masks = icenet.data.sic.mask:main",

            "icenet_process_cmip = icenet.data.processors.cmip:main",
            "icenet_process_era5 = icenet.data.processors.era5:main",
            "icenet_process_oras5 = icenet.data.processors.oras5:main",
            "icenet_process_hres = icenet.data.processors.hres:main",
            "icenet_process_sic = icenet.data.processors.osi:main",

            "icenet_process_metadata = icenet.data.processors.meta:main",

            "icenet_process_condense = "
            "icenet.data.processors.utils:condense_main",

            "icenet_dataset_check = icenet.data.dataset:check_dataset",
            "icenet_dataset_create = icenet.data.loader:create",

            "icenet_train = icenet.model.train:main",
            "icenet_predict = icenet.model.predict:main",
            "icenet_upload_azure = icenet.process.azure:upload",
            "icenet_upload_local = icenet.process.local:upload",

            "icenet_plot_record = icenet.plotting.data:plot_tfrecord",

            "icenet_plot_forecast = icenet.plotting.forecast:plot_forecast",
            "icenet_plot_input = icenet.plotting.data:plot_sample_cli",
            "icenet_plot_sic_error = icenet.plotting.forecast:sic_error",
            "icenet_plot_sic_error_local = icenet.plotting.forecast:sic_error_local",
            "icenet_plot_bin_accuracy = "
            "icenet.plotting.forecast:binary_accuracy",
            "icenet_plot_sie_error = icenet.plotting.forecast:sie_error",
            "icenet_plot_metrics = icenet.plotting.forecast:metric_plots",
            "icenet_plot_leadtime_avg = icenet.plotting.forecast:leadtime_avg_plots",

            "icenet_video_data = icenet.plotting.video:data_cli",

            "icenet_output = icenet.process.predict:create_cf_output",
            "icenet_output_geotiff = "
            "icenet.process.forecasts:create_geotiff_output",
            "icenet_output_broadcast = "
            "icenet.process.forecasts:broadcast_main",
            "icenet_output_reproject = "
            "icenet.process.forecasts:reproject_main",

            "icenet_result_threshold = "
            "icenet.results.threshold:threshold_main"
        ],
        },
    python_requires='>=3.7, <4',
    install_requires=get_content("requirements.txt"),
    include_package_data=True,
    extras_require={
        "dev": get_content("requirements_dev.txt"),
        "docs": get_content("docs/requirements.txt"),
        },
    test_suite='tests',
    tests_require=['pytest>=3'],
    zip_safe=False,
)
