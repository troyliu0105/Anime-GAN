from .load_model import load_model_from_params
from .vis import TrainingHistory


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
