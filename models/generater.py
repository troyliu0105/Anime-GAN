from mxnet import gluon


class Fucker(gluon.nn.HybridSequential):
    def __init__(self, **kwargs):
        super(Fucker, self).__init__(**kwargs)

        # in:1 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        # out = (in - 1) * strides - 2 * padding + ks + out_padding
        with self.name_scope():
            self.add(
                # input (batch, channel, 1, 1)
                gluon.nn.Conv2DTranspose(512, kernel_size=4, strides=1, padding=0, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 4, 4)

                gluon.nn.Conv2DTranspose(512, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 8, 8)

                gluon.nn.Conv2DTranspose(256, kernel_size=4, strides=2, padding=1, use_bias=False),
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
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('sigmoid'),
                # output (batch, 3, 256, 256)
            )
