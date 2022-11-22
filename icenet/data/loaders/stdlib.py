from icenet.data.loaders.base import IceNetBaseDataLoader

"""
Python Standard Library implementations for icenet data loading

Still WIP to re-introduce alternate implementations that might work better in 
certain deployments

"""


class IceNetDataLoader(IceNetBaseDataLoader):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: https://github.com/icenet-ai/icenet/blob/cb68e5dec31d4c62d72411cbca4c6d3a0276e0f9/icenet2/data/loader.py
        raise NotImplementedError("Not yet adapted from old implementation")

    def generate(self):
        """

        """
        pass

    def generate_sample(self,
                        date: object,
                        prediction: bool = False):
        """

        :param date:
        :param prediction:
        """
        pass
