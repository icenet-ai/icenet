
from icenet2.data.interfaces.downloader import ClimateDownloader


class HRESDownloader(ClimateDownloader):
    def __init__(self):
        raise NotImplementedError("{} is not ready".format(__class__.__name__))


class SEAS5Downloader(ClimateDownloader):
    def __init__(self):
        raise NotImplementedError("{} is not ready".format(__class__.__name__))
