import logging
import requests


def esgf_search(server: str = "https://esgf-node.llnl.gov/esg-search/search",
                files_type: str = "OPENDAP",
                local_node: bool = False,
                latest: bool = True,
                project: str = "CMIP6",
                format: str = "application%2Fsolr%2Bjson",
                use_csrf: bool = False,
                **search):
    """

    Below taken from
    https://hub.binder.pangeo.io/user/pangeo-data-pan--cmip6-examples-ro965nih/lab
    and adapted slightly

    :param server:
    :param files_type:
    :param local_node:
    :param latest:
    :param project:
    :param format:
    :param use_csrf:
    :param search:
    :return:
    """
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"] = "File"
    if latest:
        payload["latest"] = "true"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        logging.debug("ESGF search URL: {}".format(url))

        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            for k in d:
                logging.debug("{}: {}".format(k, d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)
