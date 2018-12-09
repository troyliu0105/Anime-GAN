from mxnet import gluon
from mxnet.gluon.nn import HybridSequential
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import LeakyReLU
from mxnet.gluon.nn import Dense
from mxnet.gluon.nn import HybridLambda
from mxnet.gluon.nn import ReflectionPad2D


class InceptionBlock(HybridBlock):
    def __init__(self, inchannels, outchannels, **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        self.inchannels = inchannels
        self.outchannels = outchannels
        with self.name_scope():
            self.x5 = HybridSequential(prefix='x5-')
            self.x5.add(
                Conv2DTranspose(outchannels, kernel_size=5, strides=1, padding=2),
                BatchNorm(),
                Activation('relu')
            )
            self.x3 = HybridSequential(prefix='x3-')
            self.x3.add(
                Conv2DTranspose(outchannels, kernel_size=3, strides=1, padding=1),
                BatchNorm(),
                Activation('relu')
            )
            self.x1 = HybridSequential(prefix='x1-')
            self.x1.add(
                Conv2D(outchannels, kernel_size=1, strides=1),
                BatchNorm(),
                Activation('relu')
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        x5_output = self.x5(x)
        x3_output = self.x3(x)
        combine = F.concat(x, x3_output, x5_output)
        return self.x1(combine)


class GeneratorV1(gluon.nn.HybridSequential):
    def __init__(self, **kwargs):
        super(GeneratorV1, self).__init__(**kwargs)

        # in:1 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        # out = (in - 1) * strides - 2 * padding + ks + out_padding
        with self.name_scope():
            self.add(
                # input (batch, channel, 1, 1)
                gluon.nn.Conv2DTranspose(512, kernel_size=4, strides=1, padding=0, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 4, 4)

                gluon.nn.Conv2DTranspose(256, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 8, 8)

                gluon.nn.Conv2DTranspose(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 256, 16, 16)

                gluon.nn.Conv2DTranspose(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 128, 32, 32)

                gluon.nn.Conv2DTranspose(64, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 64, 64, 64)

                gluon.nn.Conv2DTranspose(64, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 64, 128, 128)

                gluon.nn.Conv2DTranspose(3, kernel_size=4, strides=2, padding=1, use_bias=False),
                # gluon.nn.BatchNorm(),
                gluon.nn.Activation('tanh'),

                # gluon.nn.Conv2D(3, kernel_size=3, strides=1, padding=1, use_bias=False),
                # gluon.nn.Activation('tanh'),
                # output (batch, 3, 256, 256)
            )


class GeneratorV2(HybridSequential):
    def __init__(self, **kwargs):
        super(GeneratorV2, self).__init__(**kwargs)
        with self.name_scope():
            self.add(
                # input (batch, channel, 1, 1)
                gluon.nn.Conv2DTranspose(512, kernel_size=4, strides=1, padding=0, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 4, 4)

                gluon.nn.Conv2DTranspose(256, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 8, 8)

                gluon.nn.Conv2DTranspose(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 256, 16, 16)

                gluon.nn.Conv2DTranspose(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 128, 32, 32)

                gluon.nn.Conv2DTranspose(64, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 64, 64, 64)

                InceptionBlock(64, 32),

                gluon.nn.Conv2DTranspose(32, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),

                InceptionBlock(32, 16),

                gluon.nn.Conv2DTranspose(3, kernel_size=4, strides=2, padding=1, use_bias=False),
                Activation('tanh')
            )


class GeneratorV3(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(GeneratorV3, self).__init__(**kwargs)

        # in:1 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        # out = (in - 1) * strides - 2 * padding + ks + out_padding
        with self.name_scope():
            self.input = HybridSequential(prefix='blk0-')
            self.input.add(
                Dense(512),
                BatchNorm(),
                Activation('relu'),

                Dense(512 * 4 * 4),
                BatchNorm(),
                Activation('relu'),
                HybridLambda(lambda F, x: F.reshape(x, shape=(-1, 32, 16, 16))),
                HybridLambda(lambda F, x: F.contrib.BilinearResize2D(data=x, height=8, width=8)),
                # out (bs, 32, 8, 8)
                Conv2D(512, kernel_size=3, strides=1, padding=1, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(512, kernel_size=3, strides=1, padding=1, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(512, kernel_size=3, strides=1, padding=1, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                HybridLambda(lambda F, x: F.contrib.BilinearResize2D(data=x, height=16, width=16)),
                # out (bs, 64, 16, 16)

                Conv2D(256, kernel_size=3, strides=1, padding=1, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(256, kernel_size=3, strides=1, padding=1, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(256, kernel_size=3, strides=1, padding=1, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                HybridLambda(lambda F, x: F.contrib.BilinearResize2D(data=x, height=32, width=32)),
                # out (bs, 64, 32, 32)

                Conv2D(128, kernel_size=5, strides=1, padding=2, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(128, kernel_size=5, strides=1, padding=2, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(128, kernel_size=5, strides=1, padding=2, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                HybridLambda(lambda F, x: F.contrib.BilinearResize2D(data=x, height=64, width=64)),
                # out (bs, 128, 64, 64)

                Conv2D(64, kernel_size=5, strides=1, padding=2, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(64, kernel_size=5, strides=1, padding=2, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                Conv2D(64, kernel_size=5, strides=1, padding=2, use_bias=False),
                BatchNorm(),
                Activation('relu'),
                HybridLambda(lambda F, x: F.contrib.BilinearResize2D(data=x, height=128, width=128)),
                # out (bs, 64, 128, 128)

                ReflectionPad2D(padding=2),
                Conv2D(32, kernel_size=5, strides=1, use_bias=False),
                Activation('relu'),
                ReflectionPad2D(padding=2),
                Conv2D(3, kernel_size=5, strides=1, use_bias=False),
                HybridLambda(lambda F, x: F.contrib.BilinearResize2D(data=x, height=256, width=256)),
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.input(x)
        return out


class GeneratorV4(gluon.nn.HybridSequential):
    def __init__(self, **kwargs):
        super(GeneratorV4, self).__init__(**kwargs)

        # in:1 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        # out = (in - 1) * strides - 2 * padding + ks + out_padding
        with self.name_scope():
            self.add(
                Dense(512 * 2 * 2),
                BatchNorm(momentum=0.8),
                Activation('relu'),
                HybridLambda(lambda F, x: F.reshape(x, shape=(-1, 512, 2, 2))),
                # input (batch, channel, 2, 2)
                gluon.nn.Conv2DTranspose(512, kernel_size=3, strides=1, padding=0, use_bias=False),
                gluon.nn.BatchNorm(momentum=0.8),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 4, 4)

                gluon.nn.Conv2DTranspose(512, kernel_size=3, strides=1, padding=0, use_bias=False),
                gluon.nn.BatchNorm(momentum=0.8),
                gluon.nn.Activation('relu'),
                gluon.nn.Conv2DTranspose(512, kernel_size=3, strides=1, padding=0, use_bias=False),
                gluon.nn.BatchNorm(momentum=0.8),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 8, 8)

                gluon.nn.Conv2DTranspose(256, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(momentum=0.8),
                gluon.nn.Activation('relu'),
                # output (batch, 256, 16, 16)

                gluon.nn.Conv2DTranspose(256, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(momentum=0.8),
                gluon.nn.Activation('relu'),
                # output (batch, 128, 32, 32)

                gluon.nn.Conv2DTranspose(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(momentum=0.8),
                gluon.nn.Activation('relu'),
                # output (batch, 64, 64, 64)

                gluon.nn.Conv2DTranspose(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(momentum=0.8),
                gluon.nn.Activation('relu'),
                # output (batch, 64, 128, 128)

                gluon.nn.Conv2DTranspose(3, kernel_size=4, strides=2, padding=1, use_bias=False),
                # gluon.nn.BatchNorm(),
                gluon.nn.Activation('tanh'),

                # gluon.nn.Conv2D(3, kernel_size=3, strides=1, padding=1, use_bias=False),
                # gluon.nn.Activation('tanh'),
                # output (batch, 3, 256, 256)
            )
