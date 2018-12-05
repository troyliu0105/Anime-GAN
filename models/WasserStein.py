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


class WasserSteinInit(Initializer):
    def _init_weight(self, name, arr):
        arr[:] = mx.nd.random_normal(0.0, 0.02, shape=arr.shape)

    def _init_beta(self, _, arr):
        arr[:] = mx.nd.zeros_like(arr)

    def _init_gamma(self, _, arr):
        arr[:] = mx.nd.random_normal(1.0, 0.02, shape=arr.shape)


def clip_dis(d_model: HybridBlock, clip_size):
    for param_name in d_model.collect_params():
        param = d_model.collect_params(param_name)[param_name]
        cliped_param = mx.nd.clip(param.data(), a_min=-clip_size, a_max=clip_size)
        param.set_data(cliped_param)


def wasser_penalty(dis_model, real, fake, penalty_rate, ctx=None):
    from mxnet import autograd
    alpha = mx.nd.random_uniform(shape=real.shape)
    if ctx:
        alpha.as_in_context(ctx)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates.attach_grad()
    with autograd.predict_mode():
        with autograd.record():
            z = dis_model(interpolates)
        autograd.grad(z, [interpolates], head_grads=[z], retain_graph=True)
    gradients = interpolates.grad
    gradients_penalty = mx.nd.mean(mx.nd.array(
        [(g.norm() ** 2 - 1).asscalar() for g in gradients]
    )) * penalty_rate
    if ctx:
        gradients_penalty = gradients_penalty.as_in_context(ctx)
    return gradients_penalty
