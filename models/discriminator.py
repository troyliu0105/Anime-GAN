from mxnet import gluon


class Sucker(gluon.nn.HybridSequential):
    def __init__(self, **kwargs):
        super(Sucker, self).__init__(**kwargs)

        # out = (in - ks) * strides - 2 * padding + ks + out_padding
        with self.name_scope():
            self.add(
                gluon.nn.Conv2D(64, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 64, 128, 128)

                gluon.nn.Conv2D(64, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 64, 64, 64)

                gluon.nn.Conv2D(128, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 128, 32, 32)

                gluon.nn.Conv2D(256, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 256, 16, 16)

                gluon.nn.Conv2D(512, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 8, 8)

                gluon.nn.Conv2D(512, kernel_size=4, strides=2, padding=1, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 512, 4, 4)

                gluon.nn.Conv2D(1, kernel_size=4, strides=2, padding=0, use_bias=False),
                gluon.nn.BatchNorm(),
                gluon.nn.Activation('relu'),
                # output (batch, 2, 1, 1)
            )
