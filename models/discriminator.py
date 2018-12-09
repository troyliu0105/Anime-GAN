from mxnet import gluon
from mxnet.gluon.nn import HybridSequential
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import LeakyReLU
from mxnet.gluon.nn import Dense
from mxnet.gluon.nn import MaxPool2D
from mxnet.gluon.nn import HybridLambda
from mxnet.gluon.nn import ReflectionPad2D
from mxnet.gluon.nn import GlobalAvgPool2D
from .layers.res import ResidualBlock


class Discriminator(gluon.nn.HybridSequential):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        # out = (in - ks) * strides - 2 * padding + ks + out_padding
        with self.name_scope():
            self.add(
                gluon.nn.Conv2D(32, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.2),
                gluon.nn.Dropout(0.25),
                # output (batch, 64, 128, 128)

                gluon.nn.Conv2D(64, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.2),
                gluon.nn.Dropout(0.25),
                # output (batch, 64, 64, 64)

                gluon.nn.Conv2D(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.2),
                gluon.nn.Dropout(0.25),
                # output (batch, 128, 32, 32)

                gluon.nn.Conv2D(256, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.2),
                gluon.nn.Dropout(0.25),
                # output (batch, 256, 16, 16)

                gluon.nn.Conv2D(512, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.2),
                gluon.nn.Dropout(0.25),
                # output (batch, 512, 8, 8)

                gluon.nn.Conv2D(512, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.2),
                gluon.nn.Dropout(0.25),
                # output (batch, 512, 4, 4)

                gluon.nn.Conv2D(1, kernel_size=4, strides=2, padding=0, use_bias=False),
                # gluon.nn.BatchNorm(),
                # gluon.nn.Activation('relu'),
                # output (batch, 2, 1, 1)
            )


class ResDiscriminator(gluon.nn.HybridSequential):
    def __init__(self, **kwargs):
        super(ResDiscriminator, self).__init__(**kwargs)

        # out = (in - ks) * strides - 2 * padding + ks + out_padding
        with self.name_scope():
            self.add(
                Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False),
                BatchNorm(),
                LeakyReLU(0.2),
                MaxPool2D(pool_size=(2, 2)),
                # out (bs, 64, 64, 64)

                ResidualBlock(64, in_channels=64, downsample=False),
                ResidualBlock(64, in_channels=64, downsample=False),
                # out (bs, 64, 32, 32)

                ResidualBlock(128, in_channels=64, strides=(2, 1), downsample=True),
                ResidualBlock(128, in_channels=128, downsample=False),
                # out (bs, 128, 16, 16)

                ResidualBlock(256, in_channels=128, strides=(2, 1), downsample=True),
                ResidualBlock(256, in_channels=256, downsample=False),
                # out (bs, 256, 8, 8)

                ResidualBlock(512, in_channels=258, strides=(2, 1), downsample=True),
                ResidualBlock(512, in_channels=512, downsample=False),
                # out (bs, 512, 4, 4)
                BatchNorm(),
                LeakyReLU(0.2),
                GlobalAvgPool2D(),
                Dense(128),
                LeakyReLU(0.2),
                Dense(1)
            )
