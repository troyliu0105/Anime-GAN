import os
from matplotlib import pyplot as plt
from mxnet import nd
import numpy as np


def _last_save_num(save_path):
    items = sorted([item.name for item in os.scandir(save_path) if item.name.endswith('jpg')])
    if len(items) == 0:
        return 0
    return int(items[-1].split('.')[0])


_saved_times = 0


def trans_array_to_image(arr: nd.NDArray):
    arr = arr * 0.5 + 0.5
    arr *= 255
    arr = nd.clip(arr, a_min=0, a_max=255)
    return arr.asnumpy().astype('uint8')


def show_img(img_arr, title=None, save_path=None):
    global _saved_times
    # check whether need to get last save_num
    if _saved_times == 0:
        _saved_times = _last_save_num(save_path) + 1
    img = trans_array_to_image(img_arr)
    plt.clf()
    if title:
        plt.title(title)
    if save_path:
        plt.imsave(os.path.join(save_path, "{:04d}.jpg".format(_saved_times)), img)
        _saved_times += 1
    else:
        plt.imshow(img)
        plt.show()
