import hashlib
import os

import requests
import tqdm
from mxnet.gluon.data.vision import ImageRecordDataset

from .downloader import sess

dataset_center = os.path.expanduser("~/.mxnet/datasets")
if not os.path.exists(dataset_center):
    os.makedirs(dataset_center)


def _download(url, digest):
    name = os.path.split(url)[1]
    save_path = os.path.join(dataset_center, name)
    if os.path.exists(save_path):
        return save_path
    else:
        save_path += '.part'
    res = sess.get(url, stream=True)
    total_size = int(res.headers.get('content-length', 0))
    buf_size = 1024
    md5 = hashlib.md5()
    progress = tqdm.tqdm(desc="save {} in {}".format(name, dataset_center),
                         total=total_size, unit='B', mininterval=0.25, maxinterval=1.0,
                         unit_scale=True, leave=True, dynamic_ncols=True)
    with open(save_path, mode='wb') as f:
        for data in res.iter_content(chunk_size=buf_size):
            md5.update(data)
            f.write(data)
            progress.update(len(data))
    if md5.hexdigest() != digest:
        raise IOError('md5 mismatch!')
    else:
        complete_path = os.path.splitext(save_path)[0]
        os.rename(save_path, complete_path)
    return complete_path


def _make_datasets(data_desc):
    train, val = None, None
    if 'train' in data_desc:
        # download train set
        train_urls = data_desc['train']
        train_path = _download(train_urls['rec'][0], train_urls['rec'][1])
        _download(train_urls['idx'][0], train_urls['idx'][1])
        _download(train_urls['lst'][0], train_urls['lst'][1])
        train = ImageRecordDataset(train_path, flag=1)
    if 'val' in data_desc:
        # download validation set
        val_urls = data_desc['val']
        val_path = _download(val_urls['rec'][0], val_urls['rec'][1])
        _download(val_urls['idx'][0], val_urls['idx'][1])
        _download(val_urls['lst'][0], val_urls['lst'][1])
        val = ImageRecordDataset(val_path, flag=1)
    return train, val


def load_face():
    """
    shape is (256, 256, 3)

    :return: train_dataset, val_dataset
    """
    data_desc = {
        'train': {
            'rec': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/all_face_dataset_train.rec',
                '68f3ec13d12904476346ff88ec1ecf1f'
            ),
            'idx': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/all_face_dataset_train.idx',
                '30c336b8aed875e4b633d02949390bcf'
            ),
            'lst': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/all_face_dataset_train.lst',
                '54384ac4a2ae209529c61a6518ac11a9'
            )
        },
        'val': {
            'rec': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/all_face_dataset_val.rec',
                '3181a8f317178c625389ef15bff6990a'
            ),
            'idx': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/all_face_dataset_val.idx',
                '216361768b1e3bb5c516516b1c24b24b'
            ),
            'lst': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/all_face_dataset_val.lst',
                'a787b8ee9c6c8bff720ca0084897b776'
            )
        }
    }
    return _make_datasets(data_desc)


def load_miku():
    """
    shape is (512, 512, 3)

    :return: train_dataset, val_dataset
    """
    data_desc = {
        'train': {
            'rec': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/miku_dataset_train.rec',
                '1024412d1d5ea1c45a39e57a3a9b9980'
            ),
            'idx': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/miku_dataset_train.idx',
                '738adec0c661b8d74d0687f1ebe3f02a'
            ),
            'lst': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/miku_dataset_train.lst',
                '492b0663b4dbcffaedbf637646443b0e'
            )
        },
        'val': {
            'rec': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/miku_dataset_val.rec',
                '528a494dea2c64a14a44715b417ddf4f'
            ),
            'idx': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/miku_dataset_val.idx',
                'dea75c53d1bbcdf384036b21094959eb'
            ),
            'lst': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/miku_dataset_val.lst',
                'd3edef0fb1bedcf049311827fc351ee5'
            )
        }
    }
    return _make_datasets(data_desc)


def load_rem():
    """
    shape is (256, 256, 3)

    :return: train_dataset, val_dataset
    """
    data_desc = {
        'train': {
            'rec': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/rem_dataset_train.rec',
                'cbecdfea057ebeddaaafdba554047537'
            ),
            'idx': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/rem_dataset_train.idx',
                'aba5c91aa62f8e73ac3795343e20e242'
            ),
            'lst': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/rem_dataset_train.lst',
                'efb16f6cec5630c365ef82bc9daee04a'
            )
        },
        'val': {
            'rec': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/rem_dataset_val.rec',
                'e34aa60aeb058d97becb3963e1f49984'
            ),
            'idx': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/rem_dataset_val.idx',
                '75f73a08621e235359e7c5f28bea7147'
            ),
            'lst': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/pics/rem_dataset_val.lst',
                '6171d5408e5d2abc0ae263bc655b41e7'
            )
        }
    }
    return _make_datasets(data_desc)


def load_rem_face():
    data_desc = {
        'train': {
            'rec': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/rem_face_dataset.rec',
                '521ba239f72547b4fe62af9edb8c7893'
            ),
            'idx': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/rem_face_dataset.idx',
                '73a5e4b7f0eaaaf77bfcb30af8f2c321'
            ),
            'lst': (
                'https://gitlab.com/troyliu0105/datasetrepo/raw/master/gan/face/rem_face_dataset.lst',
                'c7585ac8349b6ecc6d85259d22ad27a4'
            )
        }
    }
    return _make_datasets(data_desc)
