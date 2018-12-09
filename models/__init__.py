from .generater import GeneratorV1
from .generater import GeneratorV2
from .generater import GeneratorV3
from .generater import GeneratorV4
from .discriminator import Discriminator
from .discriminator import ResDiscriminator
from mxnet.gluon import nn


def make_gen(version: str):
    if version.lower() == 'v1':
        return GeneratorV1(prefix='generator')
    elif version.lower() == 'v2':
        return GeneratorV2(prefix='generator')
    elif version.lower() == 'v3':
        return GeneratorV3(prefix='generator')
    elif version.lower() == 'v4':
        return GeneratorV4(prefix='generator')


def make_dis(version: str = 'v1'):
    if version.lower() == 'v1':
        return Discriminator(prefix='discriminator')
    elif version.lower() == 'v2':
        return ResDiscriminator(prefix='discriminator')


def make_generic_gen(isize, nz, nc, ngf, n_extra_layers=0):
    assert isize % 16 == 0, 'isize has to be a multiple of 16'
    gen = nn.HybridSequential(prefix='generator')

    cngf, tisize = ngf // 2, 4
    while tisize != isize:
        cngf *= 2
        tisize *= 2
    # add input z vector
    gen.add(
        nn.Conv2DTranspose(in_channels=nz, channels=cngf, kernel_size=4,
                           strides=1, padding=0, use_bias=False, prefix='tconv_{}->{}-'.format(nz, cngf)),
        nn.BatchNorm(in_channels=cngf, prefix='bn_{}->{}-'.format(nz, cngf)),
        nn.LeakyReLU(0.1, prefix='lrelu{}->{}-'.format(nz, cngf))
    )
    csize, cndf = 4, cngf
    while csize < isize // 2:
        gen.add(
            nn.Conv2DTranspose(in_channels=cngf, channels=cngf // 2, kernel_size=4,
                               strides=2, padding=1, use_bias=False),
            nn.BatchNorm(in_channels=cngf // 2),
            nn.LeakyReLU(0.1)
        )
        cngf //= 2
        csize *= 2

    for extra in range(n_extra_layers):
        gen.add(
            nn.Conv2D(in_channels=cngf, channels=cngf, kernel_size=3, strides=1, padding=1, use_bias=False),
            nn.BatchNorm(in_channels=cngf),
            nn.LeakyReLU(0.1)
        )
    gen.add(
        nn.Conv2DTranspose(in_channels=cngf, channels=nc, kernel_size=4, strides=2, padding=1, use_bias=False),
        nn.Activation('tanh')
    )
    return gen


def make_generic_dis(isize, nc, ndf, n_extra_layers=0):
    assert isize % 16 == 0, 'isize has to be a multiple of 16'
    dis = nn.HybridSequential(prefix='discriminator')
    dis.add(
        nn.Conv2D(in_channels=nc, channels=ndf, kernel_size=4, strides=2, padding=1, use_bias=False),
        nn.LeakyReLU(0.2)
    )
    csize, cndf = isize // 2, ndf
    for extra in range(n_extra_layers):
        dis.add(
            nn.Conv2D(in_channels=cndf, channels=cndf, kernel_size=3, strides=1, padding=1, use_bias=False),
            nn.BatchNorm(in_channels=cndf),
            nn.LeakyReLU(0.2)
        )
    while csize > 4:
        in_features = cndf
        out_features = cndf * 2
        dis.add(
            nn.Conv2D(in_channels=in_features, channels=out_features, kernel_size=4, strides=2, padding=1,
                      use_bias=False),
            nn.BatchNorm(in_channels=out_features),
            nn.LeakyReLU(0.2)
        )
        cndf *= 2
        csize //= 2
    dis.add(
        nn.Conv2D(in_channels=cndf, channels=1, kernel_size=4, strides=1, padding=0, use_bias=False)
    )
    return dis
