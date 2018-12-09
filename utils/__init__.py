from .load_model import load_model_from_params
from .vis import TrainingHistory


def get_cpus():
    import platform
    if platform.uname().system == 'Darwin':
        import multiprocessing
        return multiprocessing.cpu_count() // 2
    else:
        return 0
