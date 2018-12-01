from .generater import Fucker
from .discriminator import Sucker


def make_fucker():
    return Fucker(prefix='generater')


def make_sucker():
    return Sucker(prefix='discriminator')
