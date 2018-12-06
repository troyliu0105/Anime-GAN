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


def weight_init(layers):
    for layer in layers:
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            layer.weight.set_data(mx.ndarray.random.normal(0.0, 0.02, shape=layer.weight.data().shape))
        elif classname.find('BatchNorm') != -1:
            layer.gamma.set_data(mx.ndarray.random.normal(1.0, 0.02, shape=layer.gamma.data().shape))
            layer.beta.set_data(mx.ndarray.zeros(shape=layer.beta.data().shape))


def clip_dis(d_model: HybridBlock, clip_size):
    for param_name in d_model.collect_params():
        param = d_model.collect_params(param_name)[param_name]
        cliped_param = mx.nd.clip(param.data(), a_min=-clip_size, a_max=clip_size)
        param.set_data(cliped_param)


def wasser_penalty(dis_model, real, fake, penalty_rate, ctx=None):
    from mxnet import autograd
    with autograd.pause():
        alpha = mx.nd.random_uniform(shape=real.shape)
        if ctx:
            alpha.as_in_context(ctx)
        interpolates = alpha * real.detach() + ((1 - alpha) * fake.detach())

    interpolates = interpolates.detach()
    interpolates.attach_grad()
    z = dis_model(interpolates)
    gradients = autograd.grad(heads=z, variables=interpolates,
                              head_grads=mx.nd.ones(shape=z.shape, ctx=ctx),
                              retain_graph=True, create_graph=True)[0]
    gradients = gradients.reshape((gradients.shape[0], -1))
    gradients_penalty = ((gradients.norm(2, axis=1) - 1) ** 2).mean() * penalty_rate
    gradients_penalty.attach_grad()
    if ctx:
        gradients_penalty = gradients_penalty.as_in_context(ctx)
    return gradients_penalty
