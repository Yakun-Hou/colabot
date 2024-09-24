import os

# ceph
# from petrel_client.client import Client
from io import BytesIO, StringIO
# client = Client('~/petreloss.conf')


def listdir(dir):
    if 's3://' in dir:
        return client.list(dir)
    else:
        return os.listdir(dir)


def isdir(dir):
    if 's3://' in dir:
        return client.isdir(dir)
    else:
        return os.path.isdir(dir)
