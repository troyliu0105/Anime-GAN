from matplotlib import pyplot as plt
from mxnet import nd
import os

_saved_times = 0


def trans_array_to_image(arr: nd.NDArray):
    arr = arr * 255
    return arr.asnumpy().astype('uint8')


def show_img(img_arr, title=None, save_path=None):
    global _saved_times
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
