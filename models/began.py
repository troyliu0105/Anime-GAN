from mxnet.gluon import nn
import numpy as np


def decoder(nz, outSize, channel=3, nf=128):
    init_dim = 8
    layers = int(np.log2(outSize) - 3)

    decoder_seq = nn.HybridSequential(prefix='decoder')
    decoder_seq.add(
        nn.Dense(nf * init_dim ** 2, in_units=nz),
        nn.HybridLambda(lambda F, x: F.reshape(x, shape=(-1, nf, init_dim, init_dim))),
        nn.Conv2D(nf, kernel_size=3, strides=3, padding=init_dim),
        nn.ELU(),
        nn.Conv2D(nf, kernel_size=3, strides=3, padding=init_dim),
        nn.ELU(),
    )
    current_dim = init_dim
    for i in range(layers):
        current_dim *= 2
        decoder_seq.add(
            nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')),
            nn.Conv2D(nf, kernel_size=3, strides=3, padding=current_dim),
            nn.ELU(),
            nn.Conv2D(nf, kernel_size=3, strides=3, padding=current_dim),
            nn.ELU(),
        )
    decoder_seq.add(
        nn.Conv2D(channel, kernel_size=3, strides=3, padding=current_dim),
        nn.ELU(),
    )
    return decoder_seq


def encoder(nz, inSize, channel=3, nf=128):
    init_dim = 8
    layers = int(np.log2(inSize) - 2)
    encoder_seq = nn.HybridSequential(prefix='encoder')
    encoder_seq.add(
        nn.Conv2D(channel, kernel_size=3, strides=3, padding=inSize),
        nn.ELU(),
    )
    current_dim = inSize
    for i in range(1, layers):
        encoder_seq.add(
            nn.Conv2D(i * nf, kernel_size=3, strides=3, padding=current_dim),
            nn.ELU(),
            # nn.Conv2D(i * nf, kernel_size=3, strides=3, padding=current_dim),
            # nn.ELU(),
            nn.Conv2D(i * nf, kernel_size=3, strides=2, padding=1),
            nn.ELU(),
        )
        current_dim //= 2
    encoder_seq.add(
        nn.Conv2D(layers * nf, kernel_size=3, strides=3, padding=current_dim),
        nn.ELU(),
        nn.Conv2D(layers * nf, kernel_size=3, strides=3, padding=current_dim),
        nn.ELU(),
    )
    encoder_seq.add(
        nn.HybridLambda(lambda F, x: F.reshape(x, shape=(-1, layers * nf * init_dim ** 2))),
        nn.Dense(nz)
    )
    return encoder_seq


def autoencoder(nz, outSize, channel, nf=128):
    autoencoder_seq = nn.HybridSequential(prefix='autoencoder')
    autoencoder_seq.add(
        encoder(nz, outSize, channel, nf),
        decoder(nz, outSize, channel, nf)
    )
    return autoencoder_seq
