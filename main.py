# %% import libs
import os
import logging
import mxnet as mx
import tqdm
from mxnet import autograd
from mxnet import gluon
from gluoncv.utils import TrainingHistory

from datasets import load_face, load_rem
from utils import vis
import models

mx.random.seed(5)
logger = logging.getLogger(name='GAN')

# %% define parameters
epoch = 10
batch_size = 16
lr = 0.1
CTX = mx.gpu()
try:
    _ = mx.nd.ones(shape=(1), ctx=CTX)
except mx.MXNetError:
    CTX = mx.cpu_pinned()
    logger.warning("Can't use gpu, use {} instead".format(CTX))
else:
    logger.info("Will use {}".format(CTX))

# %% define dataloader
logger.info("Prepare data")
tfs_train = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(256, 256), interpolation=1),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.RandomSaturation(0.05),
    gluon.data.vision.transforms.ToTensor()
])

tfs_val = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(256, 256), interpolation=1),
    gluon.data.vision.transforms.ToTensor()
])

train_set, val_set = load_rem()
train_loader = gluon.data.DataLoader(train_set.transform_first(tfs_train), batch_size=batch_size, shuffle=True,
                                     last_batch='rollover', num_workers=2, pin_memory=True)
val_loader = gluon.data.DataLoader(val_set.transform_first(tfs_val),
                                   batch_size=batch_size, shuffle=False,
                                   last_batch='rollover', num_workers=2, pin_memory=True)

# %% define models
generator = models.make_gen()
discriminator = models.make_dis()
generator.initialize(ctx=CTX)
discriminator.initialize(ctx=CTX)
generator.hybridize()
discriminator.hybridize()

# %% prepare training
history_labels = ['gloss', 'gval_loss', 'dloss', 'dval_loss']
history = TrainingHistory(labels=history_labels)
logger.info("Prepare training")
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer_gen = gluon.Trainer(generator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr,
    'wd': 0.00001
})
trainer_dis = gluon.Trainer(discriminator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr,
    'wd': 0.00001
})
true_label = mx.nd.ones((batch_size,), ctx=CTX)
fake_label = mx.nd.zeros((batch_size,), ctx=CTX)

make_noises = lambda bs: mx.nd.random_normal(0, 1, shape=(bs, 128, 1, 1), ctx=CTX, dtype='float32')


def validation(g, d, val_loader):
    g_val_loss = 0.0
    d_val_loss = 0.0
    iter_times = 0
    for data, _ in tqdm.tqdm(
            val_loader,
            desc="Validating",
            leave=True,
            unit='batch',
            unit_scale=True,
            mininterval=1,
            maxinterval=5,
            dynamic_ncols=True):
        iter_times += 1
        bs = len(data)
        nosise = make_noises(bs)
        with autograd.predict_mode():
            # loss for d
            out = d(data).flatten()
            err2real = loss(out, true_label)

            fake_img = g(nosise)
            out = d(fake_img).flatten()
            err2fake = loss(out, fake_label)

            err4sucker = err2real + err2fake
            d_val_loss += err4sucker.mean().asscalar()

            # loss for g
            fake_img = g(nosise)
            out = d(fake_img)
            err4fucker = loss(out, true_label)
            g_val_loss += err4fucker.mean().asscalar()
    return g_val_loss / iter_times, d_val_loss / iter_times


# %% begin training
logger.info("Begin training")
for ep in range(1, epoch + 1):
    g_train_loss = 0.0
    d_train_loss = 0.0
    iter_times = 0
    for data, _ in tqdm.tqdm(
            train_loader,
            desc="Epoch {}".format(ep),
            leave=True,
            unit='batch',
            unit_scale=True,
            mininterval=1,
            maxinterval=5,
            dynamic_ncols=True):
        iter_times += 1
        bs = len(data)
        nosise = make_noises(bs)

        # begin training discriminator
        with autograd.record():
            # train with real image
            out = discriminator(data)
            err2real = loss(out, true_label)

            # train with fake image
            fake_img = generator(nosise)
            out = discriminator(fake_img).flatten()
            err2fake = loss(out, fake_label)

            err4dis = err2real + err2fake
        err4dis.backward()
        trainer_dis.step(bs)
        d_train_loss += err4dis.mean().asscalar()

        # begin training generator
        with autograd.record():
            fake_img = generator(nosise)
            out = discriminator(fake_img).flatten()
            err4gen = loss(out, true_label)
        err4gen.backward()
        trainer_gen.step(bs)
        g_train_loss += err4gen.mean().asscalar()

    g_train_loss /= iter_times
    d_train_loss /= iter_times
    g_val_loss, d_val_loss = validation(generator, discriminator, val_loader)
    history.update([g_train_loss, g_val_loss, d_train_loss, d_val_loss])
    logger.info("Generator[train: {}, val: {}]".format(g_train_loss, g_val_loss))
    logger.info("Discriminator[train: {}, val: {}]".format(d_train_loss, d_val_loss))
    # if ep % 10 == 0:
    fake = generator(make_noises(1))[0]
    vis.show_img(fake.transpose((1, 2, 0)), save_path='logs/pred')

generator.save_parameters('g_params')
history.plot(history_labels, save_path='logs/historys')
