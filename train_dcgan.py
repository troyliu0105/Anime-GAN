# %% import libs
import os
import argparse
import logging as logger
import mxnet as mx
import tqdm
from mxnet import autograd
from mxnet import gluon
from gluoncv.utils import makedirs

import datasets as gan_datasets
from utils import vis, get_cpus, TrainingHistory
import models

mx.random.seed(5)
logger.basicConfig(level=logger.INFO, filename='logs/train_loss-dcgan.log')

arg = argparse.ArgumentParser(description="training parameters")
arg.add_argument('--lr', type=float, default=0.001, help='learning rate')
arg.add_argument('--batch', type=int, default=32, help='batch size')
arg.add_argument('--epoch', type=int, default=30000, help='training epochs')
arg.add_argument('--continue', type=bool, default=True, help='should continue with last checkpoint')
arg.add_argument('--save_checkpoint', type=bool, default=True, help='whether save checkpoint')
arg.add_argument('--save_per_epoch', type=int, default=250, help='save checkpoint every specific epochs')
arg.add_argument('--save_dir', type=str, default='saved/params-dcgan', help='check point save path')
arg.add_argument('--cuda', type=bool, default=False, help='whether use gpu, default is True')
arg.add_argument('--pred_per_gen', type=int, default=15, help='make a pred every specific epoch')
arg.add_argument('--validation', type=bool, default=False, help='whether use validation set, default: False')
arg.add_argument('--dataset', type=str, default='rem_face', help='rem, miku, face，rem_face')

opt = arg.parse_args()

# %% define parameters
epoch = opt.epoch
epoch_start = 0
batch_size = opt.batch
lr = opt.lr
should_save_checkpoint = opt.save_checkpoint
save_per_epoch = opt.save_per_epoch
save_dir = opt.save_dir
pred_per_epoch = opt.pred_per_epoch
should_use_val = opt.validation
dataset = opt.dataset
dataset_loader = getattr(gan_datasets, 'load_{}'.format(dataset))

CTX = mx.gpu() if opt.cuda else mx.cpu()
logger.info('Will use {}'.format(CTX))

# %% define dataloader
logger.info("Prepare data")
# noinspection PyTypeChecker
tfs_train = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(256, 256), interpolation=2),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.RandomSaturation(0.005),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# noinspection PyTypeChecker
tfs_val = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(256, 256), interpolation=2),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_set, val_set = dataset_loader()
train_loader = gluon.data.DataLoader(train_set.transform_first(tfs_train),
                                     batch_size=batch_size, shuffle=True,
                                     last_batch='rollover', num_workers=get_cpus(), pin_memory=True)
if val_set:
    val_loader = gluon.data.DataLoader(val_set.transform_first(tfs_val),
                                       batch_size=batch_size, shuffle=False,
                                       last_batch='rollover', num_workers=get_cpus(), pin_memory=True)

# %% define models
generator = models.make_gen('v4')
discriminator = models.make_dis()
generator.initialize(init=mx.init.Normal(0.02), ctx=CTX)
discriminator.initialize(init=mx.init.Normal(0.02), ctx=CTX)
if getattr(opt, 'continue'):
    import utils

    makedirs(save_dir)
    epoch_start = utils.load_model_from_params(generator, discriminator, save_dir)
    logger.info('Continue training at {}, and rest epochs {}'.format(epoch_start, epoch - epoch_start))

generator.hybridize()
discriminator.hybridize()

# %% prepare training
logger.info("Prepare training")
if should_use_val:
    history_labels = ['gloss', 'gval_loss', 'dloss', 'dval_loss']
else:
    history_labels = ['gloss', 'dloss']
history = TrainingHistory(labels=history_labels)
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
trainer_gen = gluon.Trainer(generator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr,
    'beta1': 0.5
})
trainer_dis = gluon.Trainer(discriminator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr,
    'beta1': 0.5
})
true_label = mx.nd.ones((batch_size,), ctx=CTX)
fake_label = mx.nd.zeros((batch_size,), ctx=CTX)


def make_noises(bs):
    return mx.nd.random_normal(0, 1, shape=(bs, 512), ctx=CTX, dtype='float32').reshape((bs, 512, 1, 1))


pred_noise = make_noises(1)
mx.nd.save('pred_noise', pred_noise)


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
        data = data.as_in_context(CTX)
        with autograd.predict_mode():
            # loss for d
            out = d(data)
            err2real = loss(out, true_label)

            fake_img = g(nosise)
            out = d(fake_img)
            err2fake = loss(out, fake_label)

            err4dis = err2real + err2fake
            d_val_loss += err4dis.mean().asscalar()

            # loss for g
            fake_img = g(nosise)
            out = d(fake_img)
            err4gen = loss(out, true_label)
            g_val_loss += err4gen.mean().asscalar()
    return g_val_loss / iter_times, d_val_loss / iter_times


# %% begin training
d_iter_times = 0
g_iter_times = 0
d_update_times = 0
g_update_times = 0
g_train_loss = 0.0
d_train_loss = 0.0
logger.info("Begin training")
for ep in tqdm.tqdm(range(epoch_start, epoch + 1),
                    total=epoch,
                    desc="Total Progress",
                    leave=False,
                    initial=epoch_start,
                    unit='epoch',
                    unit_scale=True,
                    mininterval=10,
                    maxinterval=100,
                    dynamic_ncols=True):

    for data, _ in tqdm.tqdm(
            train_loader,
            desc="Epoch {}".format(ep),
            leave=False,
            unit='batch',
            unit_scale=True,
            mininterval=1,
            maxinterval=5,
            dynamic_ncols=True):
        bs = len(data)
        nosise = make_noises(bs)
        data = data.as_in_context(CTX)
        # begin training discriminator
        with autograd.record():
            d_iter_times += 1
            d_update_times += 1
            # train with real image
            out = discriminator(data)
            err2real = loss(out, true_label)

            # train with fake image
            # detach the input, or its gradients will be computed
            with autograd.predict_mode():
                fake_img = generator(nosise)
            out = discriminator(fake_img.detach())
            err2fake = loss(out, fake_label)

            err4dis = err2real + err2fake
        err4dis.backward()
        trainer_dis.step(bs)
        d_train_loss += err4dis.mean().asscalar()

        if d_iter_times % 5 == 0:
            g_iter_times += 1
            g_update_times += 1
            # begin training generator
            with autograd.record():
                fake_img = generator(nosise)
                with autograd.predict_mode():
                    out = discriminator(fake_img)
                err4gen = loss(out, true_label)
            err4gen.backward()
            trainer_gen.step(bs)
            g_train_loss += err4gen.mean().asscalar()

            g_train_loss /= d_iter_times
            d_train_loss /= g_iter_times

            # use validation set or not
            if should_use_val:
                g_val_loss, d_val_loss = validation(generator, discriminator, val_loader)
                history.update([g_train_loss, g_val_loss, d_train_loss, d_val_loss])
                logger.info("Generator[train: {}, val: {}]".format(g_train_loss, g_val_loss))
                logger.info("Discriminator[train: {}, val: {}]".format(d_train_loss, d_val_loss))
            else:
                history.update([g_train_loss, d_train_loss])
                logger.info("Generator[{}], Discriminator[{}]".format(g_train_loss, d_train_loss))

            g_train_loss = 0.0
            d_train_loss = 0.0
            d_iter_times = 0
            g_iter_times = 0

            # make a prediction
            if g_update_times % pred_per_epoch == 0:
                fake = generator(make_noises(1))[0]
                unique_fake = generator(pred_noise)[0]
                pred_path = 'logs/pred-dcgan'
                pred_unique_path = os.path.join(pred_path, 'unique')
                makedirs(pred_path)
                makedirs(pred_unique_path)
                vis.show_img(fake.transpose((1, 2, 0)), save_path=pred_path)
                vis.show_img(unique_fake.transpose((1, 2, 0)), save_path=pred_unique_path)

                # save history plot every epoch
                history.plot(save_path='logs/histories-dcgan')

    # save checkpoint
    if should_save_checkpoint:
        if ep % save_per_epoch == 0:
            generator.save_parameters(os.path.join(save_dir, 'generator_{:04d}.params'.format(ep)))
            discriminator.save_parameters(os.path.join(save_dir, 'discriminator_{:04d}.params'.format(ep)))

history.plot(save_path='logs/histories-dcgan')
generator.save_parameters(os.path.join(save_dir, 'generator_{:04d}.params'.format(ep)))
