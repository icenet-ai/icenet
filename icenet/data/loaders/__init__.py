import inspect

from icenet.data.loaders.base import IceNetBaseDataLoader

import icenet.data.loaders.dask
import icenet.data.loaders.stdlib


class IceNetDataLoaderFactory:
    """

    """
    def __init__(self):
        self._loader_map = dict(
            dask=icenet.data.loaders.dask.DaskMultiWorkerLoader,
            dask_shared=icenet.data.loaders.dask.DaskMultiSharingWorkerLoader,
            standard=icenet.data.loaders.stdlib.IceNetDataLoader,
        )

    def add_data_loader(self, loader_name: str, loader_impl: object):
        """

        :param loader_name:
        :param loader_impl:
        """
        if loader_name not in self._loader_map:
            if IceNetBaseDataLoader in inspect.getmro(loader_impl):
                self._loader_map[loader_name] = loader_impl
            else:
                raise RuntimeError("{} is not descended from "
                                   "IceNetBaseDataLoader".
                                   format(loader_impl.__name__))
        else:
            raise RuntimeError("Cannot add {} as already in loader map".
                               format(loader_name))

    def create_data_loader(self, loader_name, *args, **kwargs):
        """

        :param loader_name:
        :param args:
        :param kwargs:
        :return:
        """
        return self._loader_map[loader_name](*args, **kwargs)

    @property
    def loader_map(self):
        return self._loader_map
