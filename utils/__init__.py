from .load_model import load_model_from_params
from .vis import TrainingHistory
from mxnet.ndarray import random_normal


def get_cpus():
    import platform
    import multiprocessing
    system = platform.uname().system
    if system == 'Darwin':
        return multiprocessing.cpu_count() // 2
    elif system == 'Linux':
        return multiprocessing.cpu_count()
    elif system == 'Windows':
        return 0


def make_noise(bs, nz, ctx):
    return random_normal(0, 1, shape=(bs, nz, 1, 1), ctx=ctx, dtype='float32')
