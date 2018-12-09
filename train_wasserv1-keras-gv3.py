# %% import libs
import os
import argparse
import logging as logger
import mxnet as mx
import tqdm
from mxnet import autograd
from mxnet import gluon
from gluoncv.utils import TrainingHistory, makedirs

import datasets as gan_datasets
import models
from utils import vis, get_cpus
from models.WasserStein import clip_dis, wasser_penalty
from models.WasserStein import WasserSteinLoss

mx.random.seed(5)
logger.basicConfig(level=logger.INFO, filename='logs/train_loss-w1keras-gv3.log')

arg = argparse.ArgumentParser(description="training parameters")
arg.add_argument('--lr', type=float, default=0.00005, help='learning rate')
arg.add_argument('--batch', type=int, default=32, help='batch size')
arg.add_argument('--epoch', type=int, default=150000, help='training epochs')
arg.add_argument('--continue', type=bool, default=True, help='should continue with last checkpoint')
arg.add_argument('--save_checkpoint', type=bool, default=True, help='whether save checkpoint')
arg.add_argument('--save_per_epoch', type=int, default=250, help='save checkpoint every specific epochs')
arg.add_argument('--save_dir', type=str, default='saved/params-w1keras', help='check point save path')
arg.add_argument('--cuda', action='store_true', default=False, help='whether use gpu, default is True')
arg.add_argument('--pred_per_gen', type=int, default=25, help='make a pred every specific epoch')
arg.add_argument('--validation', type=bool, default=False, help='whether use validation set, default: False')
arg.add_argument('--dataset', type=str, default='rem_face', help='rem, miku, face, rem_face')

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
    gluon.data.vision.transforms.RandomSaturation(0.05),
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
generator = models.make_gen('v3')
discriminator = models.make_dis()
generator.initialize(init=mx.init.Normal(0.02), ctx=CTX)
discriminator.initialize(init=mx.init.Normal(0.02), ctx=CTX)
discriminator(mx.nd.random_uniform(shape=(1, 3, 256, 256), ctx=CTX))
generator(mx.nd.random_uniform(shape=(1, 512, 1, 1), ctx=CTX))
if getattr(opt, 'continue'):
    import utils

    makedirs(save_dir)
    epoch_start = utils.load_model_from_params(generator, discriminator, save_dir)
    logger.info('Continue training at {}, and rest epochs {}'.format(epoch_start, epoch - epoch_start))

generator.hybridize()
discriminator.hybridize()


def make_noise(bs):
    return mx.nd.random_normal(0, 1, shape=(bs, 512, 1, 1), ctx=CTX, dtype='float32')


# %% prepare training
logger.info("Prepare training")
if should_use_val:
    history_labels = ['gloss', 'gval_loss', 'dloss', 'dval_loss']
else:
    history_labels = ['gloss', 'dloss']
history = TrainingHistory(labels=history_labels)
loss = WasserSteinLoss()
# scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[100, 150, 170, 200, 300, 310, 320], factor=0.5, base_lr=lr)
trainer_gen = gluon.Trainer(generator.collect_params(), optimizer='rmsprop', optimizer_params={
    'learning_rate': lr,
    'epsilon': 1e-13,
})
trainer_dis = gluon.Trainer(discriminator.collect_params(), optimizer='rmsprop', optimizer_params={
    'learning_rate': lr,
    'epsilon': 1e-11,
    'clip_weights': 0.01
})
true_label = -mx.nd.ones((batch_size,), ctx=CTX)
fake_label = mx.nd.ones((batch_size,), ctx=CTX)

pred_noise = make_noise(1)
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
        nosise = make_noise(bs)
        data = data.as_in_context(CTX)
        with autograd.predict_mode():
            # loss for d
            err2real = d(data).mean()

            fake_img = g(nosise)
            err2fake = d(fake_img).mean()

            penalty = wasser_penalty(d, data, fake_img, 10, ctx=CTX)

            err4sucker = -(err2real - err2fake) + penalty
            d_val_loss += err4sucker.asscalar()

            # loss for g
            fake_img = g(nosise)
            err4fucker = -d(fake_img).mean()
            g_val_loss += err4fucker.asscalar()
    return g_val_loss / iter_times, d_val_loss / iter_times


# %% begin training
logger.info("Begin training")
dis_update_time = 0
gen_update_time = 0
iter4G = 150

g_train_loss = 0.0
d_train_loss = 0.0
g_iter_times = 0
d_iter_times = 0
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
        noise = make_noise(bs)
        data = data.as_in_context(CTX)
        # begin training discriminator
        if dis_update_time < iter4G:
            d_iter_times += 1
            dis_update_time += 1
            with autograd.record():
                # train with real image
                err2real = loss(discriminator(data), true_label)

                fake_img = generator(noise)
                # train with fake image
                # detach the input, or its gradients will be computed
                err2fake = loss(discriminator(fake_img.detach()), fake_label)

                err4dis = err2real + err2fake
            err4dis.backward()
            trainer_dis.step(bs)
            d_train_loss += (err4dis.asscalar() / 2)

        # clip the discriminator
        # clip_dis(discriminator, 0.01)

        # begin training generator
        if dis_update_time == iter4G:
            dis_update_time = 0
            gen_update_time += 1
            with autograd.record():
                g_iter_times += 1
                fake_img = generator(noise)
                err4gen = loss(discriminator(fake_img), true_label)
            err4gen.backward()
            trainer_gen.step(bs)
            g_train_loss += err4gen.asscalar()

            g_train_loss /= g_iter_times
            d_train_loss /= d_iter_times

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
            g_iter_times = 0
            d_iter_times = 0

            if gen_update_time > 25:
                iter4G = 5

        # make a prediction
        if ep % pred_per_epoch == 0:
            fake = generator(make_noise(1))[0]
            unique_fake = generator(pred_noise)[0]
            pred_path = 'logs/pred-w1keras'
            pred_unique_path = os.path.join(pred_path, 'unique')
            makedirs(pred_path)
            makedirs(pred_unique_path)
            vis.show_img(fake.transpose((1, 2, 0)), save_path=pred_path)
            vis.show_img(unique_fake.transpose((1, 2, 0)), save_path=pred_unique_path)

        # save checkpoint
        if should_save_checkpoint:
            if ep % save_per_epoch == 0:
                generator.save_parameters(os.path.join(save_dir, 'generator_{:04d}.params'.format(ep)))
                discriminator.save_parameters(os.path.join(save_dir, 'discriminator_{:04d}.params'.format(ep)))

        # save history plot every epoch
        history.plot(history_labels, save_path='logs/historys-w1keras')

history.plot(history_labels, save_path='logs/history-sw1keras')
