import os
from matplotlib import pyplot as plt
from mxnet import nd
import math


class TrainingHistory(object):
    def __init__(self, labels):
        self.labels = labels
        self.datas = []
        for _ in labels:
            self.datas.append([])

    def update(self, datas):
        for i in range(len(datas)):
            self.datas[i].append(datas[i])

    def plot(self, save_path=None, colors=None):
        n = len(self.datas)
        if colors is None:
            colors = ['C' + str(i) for i in range(n)]
        line_lists = []
        plt.xlabel = 'epoch'
        plt.ylabel = 'loss'
        for i in range(n):
            data = self.datas[i]
            p = plt.plot(list(range(len(data))), data, colors[i], label=self.labels[i])[0]
            line_lists.append(p)
        plt.legend(line_lists, self.labels, loc='upper right')
        if save_path:
            save_path = os.path.expanduser(save_path)
            plt.savefig(save_path)
        else:
            plt.show()


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


def show_img(images, title=None, save_path=None, epoch=None):
    row = int(math.sqrt(images.shape[0]))
    col = row
    height = sum(img.shape[0] for img in images[0:row])
    width = sum(img.shape[1] for img in images[0:col])
    output = nd.zeros(shape=(height, width, 3))
    for r in range(row):
        for c in range(col):
            img = images[r * row + c]
            h, w, d = img.shape
            output[r * h:r * h + h, c * w: c * w + w] = img
    output = trans_array_to_image(output)
    global _saved_times
    plt.clf()
    if title:
        plt.title(title)
    if save_path:
        if epoch:
            path = os.path.join(save_path, "epoch_{}.jpg".format(epoch))
        else:
            # check whether need to get last save_num
            if _saved_times == 0:
                _saved_times = _last_save_num(save_path) + 1
            path = os.path.join(save_path, "{:04d}.jpg".format(_saved_times))
            _saved_times += 1
        plt.imsave(path, output)
    else:
        plt.imshow(output)
        plt.show()
