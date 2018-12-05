import mxnet as mx
from mxnet.gluon.loss import Loss
from mxnet.gluon.nn import HybridBlock
from mxnet.initializer import Initializer


class WasserSteinLoss(Loss):
    def __init__(self, clip_size=0.01, **kwargs):
        super(WasserSteinLoss, self).__init__(weight=None, batch_axis=0, **kwargs)
        self.clip_size = clip_size

    def hybrid_forward(self, F, pred, label):
        pred = F.reshape_like(pred, label)
        return F.mean(F.elemwise_mul(pred, label))

    def clip(self, d_model: HybridBlock):
        for param_name in d_model.collect_params():
            param = d_model.collect_params(param_name)[param_name]
            cliped_param = mx.nd.clip(param.data(), a_min=-self.clip_size, a_max=self.clip_size)
            param.set_data(cliped_param)


class WasserSteinInit(Initializer):
    def _init_weight(self, name, arr):
        arr[:] = mx.nd.random_normal(0.0, 0.02, shape=arr.shape)

    def _init_beta(self, _, arr):
        arr[:] = mx.nd.zeros_like(arr)

    def _init_gamma(self, _, arr):
        arr[:] = mx.nd.random_normal(1.0, 0.02, shape=arr.shape)
