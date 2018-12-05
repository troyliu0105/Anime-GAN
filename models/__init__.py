from .generater import GeneratorV1
from .generater import GeneratorV2
from .discriminator import Discriminator


def make_gen(version: str):
    if version.lower() == 'v1':
        return GeneratorV1(prefix='generater')
    elif version.lower() == 'v2':
        return GeneratorV2(prefix='generater')


def make_dis():
    return Discriminator(prefix='discriminator')
