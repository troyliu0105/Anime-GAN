from mxnet import gluon
from mxnet.gluon.nn import HybridSequential
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose


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
