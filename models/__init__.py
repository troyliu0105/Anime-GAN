from .generater import Generator
from .discriminator import Discriminator


def make_gen():
    return Generator(prefix='generater')


def make_dis():
    return Discriminator(prefix='discriminator')
