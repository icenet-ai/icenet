"""

"""

SIC_HEMI_STR = dict(
    north="nh",
    south="sh"
)


# This is adapted from the data/loaders implementations
class DaskWrapper:
    """

    :param dask_port:
    :param dask_timeouts:
    :param dask_tmp_dir:
    :param workers:
    """

    def __init__(self,
                 dask_port: int = 8888,
                 dask_timeouts: int = 60,
                 dask_tmp_dir: object = "/tmp",
                 workers: int = 8):

        self._dashboard_port = dask_port
        self._timeout = dask_timeouts
        self._tmp_dir = dask_tmp_dir
        self._workers = workers

    def dask_process(self,
                     *args,
                     method: callable,
                     **kwargs):
        """

        :param method:
        """
        dashboard = "localhost:{}".format(self._dashboard_port)

        with dask.config.set({
            "temporary_directory": self._tmp_dir,
            "distributed.comm.timeouts.connect": self._timeout,
            "distributed.comm.timeouts.tcp": self._timeout,
        }):
            cluster = LocalCluster(
                dashboard_address=dashboard,
                n_workers=self._workers,
                threads_per_worker=1,
                scheduler_port=0,
            )
            logging.info("Dashboard at {}".format(dashboard))

            with Client(cluster) as client:
                logging.info("Using dask client {}".format(client))
                ret = method(*args, **kwargs)
        return ret
