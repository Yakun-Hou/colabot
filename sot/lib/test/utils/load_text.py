import numpy as np
import pandas as pd

# from petrel_client.client import Client
from io import BytesIO, StringIO
# client = Client('~/petreloss.conf')


def load_text_numpy(path, dtype, delimiter=None):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                if 's3://' in path:  # ceph
                    raw = client.get(path)
                    ground_truth_rect = np.loadtxt(BytesIO(raw), delimiter=d, dtype=dtype)
                else:
                    ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        if 's3://' in path:  # ceph
            raw = client.get(path)
            ground_truth_rect = np.loadtxt(BytesIO(raw), delimiter=delimiter, dtype=dtype)
        else:
            ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, dtype, delimiter):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                if 's3://' in path:  # ceph
                    raw = client.get(path)
                    ground_truth_rect = pd.read_csv(BytesIO(raw), delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                    low_memory=False).values
                else:
                    ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                    low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        if 's3://' in path:  # ceph
            raw = client.get(path)
            ground_truth_rect = pd.read_csv(BytesIO(raw), delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                            low_memory=False).values
        else:
            ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                            low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, dtype, delimiter)
    elif backend == 'pandas':
        return load_text_pandas(path, dtype, delimiter)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str
