from ecmwfapi import ECMWFService

from icenet2.data.interfaces.downloader import ClimateDownloader


class HRESDownloader(ClimateDownloader):
    def __init__(self):
        raise NotImplementedError("{} is not ready".format(__class__.__name__))

    def download(self):
        server = ECMWFService("mars")
        server.execute({
            "class": "od",
            "date": "20210909",
            "expver": "1",
            "levtype": "sfc",
            "param": "167.128",
            "step": "0-168",
            "stream": "enfo",
            "format": "netcdf",
            "time": "00",
            "type": "fc",
            "target": "output",
        },
        "target.nc")


class SEAS5Downloader(ClimateDownloader):
    def __init__(self):
        raise NotImplementedError("{} is not ready".format(__class__.__name__))
