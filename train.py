# %% import libs
import os
import argparse
import logging as logger
import mxnet as mx
import tqdm
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.data.vision import ImageRecordDataset
from gluoncv.utils import TrainingHistory

from datasets import load_face, load_rem
from utils import vis
import models

mx.random.seed(5)
logger.basicConfig(level=logger.INFO, filename='logs/train_loss.log')

arg = argparse.ArgumentParser(description="training parameters")
arg.add_argument('--lr', type=float, default=0.0001, help='learning rate')
arg.add_argument('--batch', type=int, default=8, help='batch size')
arg.add_argument('--epoch', type=int, default=500, help='training epochs')
arg.add_argument('--continue', type=bool, default=True, help='should continue with last checkpoint')
arg.add_argument('--save_checkpoint', type=bool, default=False, help='whether save checkpoint')
arg.add_argument('--save_per_epoch', type=int, default=5, help='save checkpoint every specific epochs')
arg.add_argument('--save_dir', type=str, default='saved/params', help='check point save path')
arg.add_argument('--cuda', type=bool, default=True, help='whether use gpu, default is True')
arg.add_argument('--pred_per_epoch', type=int, default=2, help='make a pred every specific epoch')
arg.add_argument('--validation', type=bool, default=False, help='whether use validation set, default: False')

opt = arg.parse_args()

# %% define parameters
epoch = opt.epoch
batch_size = opt.batch
lr = opt.lr
should_save_checkpoint = opt.save_checkpoint
save_per_epoch = opt.save_per_epoch
save_dir = opt.save_dir
pred_per_epoch = opt.pred_pre_epoch
should_use_val = opt.validation

CTX = mx.gpu() if opt.cuda else mx.cpu()
logger.info('Will use {}'.format(CTX))

# %% define dataloader
logger.info("Prepare data")
# noinspection PyTypeChecker
tfs_train = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(256, 256), interpolation=1),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # gluon.data.vision.transforms.RandomSaturation(0.05),
    # gluon.data.vision.transforms.ToTensor()
])

# noinspection PyTypeChecker
tfs_val = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(256, 256), interpolation=1),
    # gluon.data.vision.transforms.ToTensor()
])

train_set, val_set = load_rem()
rem_face_set = ImageRecordDataset('rem_face_dataset.rec')
train_loader = gluon.data.DataLoader(rem_face_set.transform_first(tfs_train),
                                     batch_size=batch_size, shuffle=True,
                                     last_batch='rollover', num_workers=4, pin_memory=True)
val_loader = gluon.data.DataLoader(val_set.transform_first(tfs_val),
                                   batch_size=batch_size, shuffle=False,
                                   last_batch='rollover', num_workers=2, pin_memory=True)

# %% define models
generator = models.make_gen()
discriminator = models.make_dis()
generator.initialize(init=mx.init.Normal(0.02), ctx=CTX)
discriminator.initialize(init=mx.init.Normal(0.02), ctx=CTX)
if getattr(opt, 'continue'):
    import utils

    utils.load_model_from_params(generator, discriminator, save_dir)

generator.hybridize()
discriminator.hybridize()

# %% prepare training
if should_use_val:
    history_labels = ['gloss', 'gval_loss', 'dloss', 'dval_loss']
else:
    history_labels = ['gloss', 'dloss']
history = TrainingHistory(labels=history_labels)
logger.info("Prepare training")
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
trainer_gen = gluon.Trainer(generator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr * 2,
    'beta1': 0.5
    # 'momentum': 0.9,
    # 'wd': 0.00001
})
trainer_dis = gluon.Trainer(discriminator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr,
    'beta1': 0.5
    # 'wd': 0.00001
})
true_label = mx.nd.ones((batch_size,), ctx=CTX)
fake_label = mx.nd.zeros((batch_size,), ctx=CTX)

make_noises = lambda bs: mx.nd.random_normal(0, 1, shape=(bs, 512, 1, 1), ctx=CTX, dtype='float32')


def validation(g, d, val_loader):
    g_val_loss = 0.0
    d_val_loss = 0.0
    iter_times = 0
    for data, _ in tqdm.tqdm(
            val_loader,
            desc="Validating",
            leave=False,
            unit='batch',
            unit_scale=True,
            mininterval=1,
            maxinterval=5,
            dynamic_ncols=True):
        iter_times += 1
        bs = len(data)
        nosise = make_noises(bs)
        data = data.as_in_context(CTX).transpose((0, 3, 1, 2)).astype('float32') / 127.5 - 1.
        with autograd.predict_mode():
            # loss for d
            out = d(data)
            err2real = loss(out, true_label)

            fake_img = g(nosise)
            out = d(fake_img)
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
for ep in tqdm.tqdm(range(1, epoch + 1),
                    desc="Totol Progress",
                    leave=False,
                    unit='epoch',
                    unit_scale=True,
                    mininterval=10,
                    maxinterval=100,
                    dynamic_ncols=True
                    ):
    g_train_loss = 0.0
    d_train_loss = 0.0
    iter_times = 0
    for data, _ in tqdm.tqdm(
            train_loader,
            desc="Epoch {}".format(ep),
            leave=False,
            unit='batch',
            unit_scale=True,
            mininterval=1,
            maxinterval=5,
            dynamic_ncols=True):
        iter_times += 1
        bs = len(data)
        nosise = make_noises(bs)
        data = data.as_in_context(CTX).transpose((0, 3, 1, 2)).astype('float32') / 127.5 - 1.
        # begin training discriminator
        with autograd.record():
            # train with real image
            out = discriminator(data)
            err2real = loss(out, true_label)

            # train with fake image
            fake_img = generator(nosise.detach())
            out = discriminator(fake_img)
            err2fake = loss(out, fake_label)

            err4dis = err2real + err2fake
        err4dis.backward()
        trainer_dis.step(bs)
        d_train_loss += err4dis.mean().asscalar()

        # begin training generator
        with autograd.record():
            fake_img = generator(nosise)
            out = discriminator(fake_img)
            err4gen = loss(out, true_label)
        err4gen.backward()
        trainer_gen.step(bs)
        g_train_loss += err4gen.mean().asscalar()

    g_train_loss /= iter_times
    d_train_loss /= iter_times

    # use validation set or not
    if should_use_val:
        g_val_loss, d_val_loss = validation(generator, discriminator, val_loader)
        history.update([g_train_loss, g_val_loss, d_train_loss, d_val_loss])
        logger.info("Generator[train: {}, val: {}]".format(g_train_loss, g_val_loss))
        logger.info("Discriminator[train: {}, val: {}]".format(d_train_loss, d_val_loss))
    else:
        history.update([g_train_loss, d_train_loss])
        logger.info("Generator[{}], Discriminator[{}]".format(g_train_loss, d_train_loss))

    # make a prediction
    if ep % pred_per_epoch == 0:
        fake = generator(make_noises(1))[0]
        vis.show_img(fake.transpose((1, 2, 0)), save_path='logs/pred')

    # save checkpoint
    if should_save_checkpoint:
        if ep % save_per_epoch == 0:
            generator.save_parameters(os.path.join(save_dir, 'generator_{:04d}.params'.format(ep)))
            discriminator.save_parameters(os.path.join(save_dir, 'discriminator_{:04d}.params'.format(ep)))

    # save history plot every epoch
    history.plot(history_labels, save_path='logs/historys')

history.plot(history_labels, save_path='logs/historys')
