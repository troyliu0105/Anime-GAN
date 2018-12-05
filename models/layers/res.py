from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.nn import HybridSequential
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import GlobalAvgPool2D
from mxnet.gluon.nn import Flatten
from mxnet.gluon.nn import Dense


class ResidualBlock(HybridBlock):
    def __init__(self, channels, in_channels, downsample=False, use_se=True, se_divide=2, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.channels = channels
        self.in_channels = in_channels
        self.use_se = use_se
        self.se_divide = se_divide

        with self.name_scope():
            self.body = HybridSequential(prefix='body-')
            with self.body.name_scope():
                self.body.add(
                    Conv2D(channels, kernel_size=3, padding=1, in_channels=in_channels),
                    BatchNorm(axis=1, in_channels=channels),
                    Activation('relu'),
                    Conv2D(channels, kernel_size=3, padding=1, in_channels=channels),
                    BatchNorm(axis=1, in_channels=channels)
                )
            if downsample:
                self.downsample = HybridSequential(prefix='downsample-')
                with self.downsample.name_scope():
                    self.downsample.add(
                        Conv2D(channels, kernel_size=1, in_channels=in_channels),
                        BatchNorm(axis=1, in_channels=channels)
                    )
            else:
                self.downsample = None
            self.out_act = HybridSequential(prefix='outact-')
            with self.out_act.name_scope():
                self.out_act.add(
                    BatchNorm(axis=1),
                    Activation('relu'),
                )

            if self.use_se:
                self.se_control = HybridSequential(prefix='se-')
                with self.se_control.name_scope():
                    self.se_control.add(
                        GlobalAvgPool2D(),
                        Flatten(),
                        Dense(channels // self.se_divide, activation='relu', prefix='squeeze-'),
                        Dense(channels, activation='sigmoid', prefix='excitation-')
                    )

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        if self.use_se:
            scale = self.se_control(x)
            x = F.broadcast_mul(x, F.reshape(scale, shape=(-1, self.channels, 1, 1)))

        return self.out_act(x + residual)
