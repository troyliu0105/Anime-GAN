# %% import libs
import os
import argparse
import logging as logger
import mxnet as mx
import tqdm
import numpy as np
from functools import partial
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.data.vision.datasets import ImageRecordDataset
from gluoncv.utils import makedirs

import datasets as gan_datasets
from models import began
from utils import vis, get_cpus, TrainingHistory, var_saver

mx.random.seed(5)
logger.basicConfig(level=logger.INFO, filename='logs/train_loss-w1.log')

arg = argparse.ArgumentParser(description="training parameters")
arg.add_argument('--lr', type=float, default=0.00005, help='learning rate')
arg.add_argument('--nz', type=int, default=256, help='z vector dimension')
arg.add_argument('--imageSize', type=int, default=256, help='image size')
arg.add_argument('--batch', type=int, default=8, help='batch size')
arg.add_argument('--epoch', type=int, default=5000000, help='training epochs')
arg.add_argument('--continue', action='store_true', default=False, help='should continue with last checkpoint')
arg.add_argument('--save_checkpoint', action='store_true', default=False, help='whether save checkpoint')
arg.add_argument('--save_per_epoch', type=int, default=500, help='save checkpoint every specific epochs')
arg.add_argument('--save_dir', type=str, default='saved/params-w1', help='check point save path')
arg.add_argument('--cuda', action='store_true', default=False, help='whether use gpu, default is True')
arg.add_argument('--pred_per_gen', type=int, default=500, help='make a pred generator update')
arg.add_argument('--validation', type=bool, default=False, help='whether use validation set, default: False')
arg.add_argument('--dataset', type=str, default='rem', help='rem, miku, face, rem_face')

opt = arg.parse_args()
print(opt)

# %% define parameters
epoch = opt.epoch
epoch_start = 0
batch_size = opt.batch
lr = opt.lr
nz = opt.nz
imageSize = opt.imageSize
should_save_checkpoint = opt.save_checkpoint
save_per_epoch = opt.save_per_epoch
save_dir = opt.save_dir
pred_per_gen = opt.pred_per_gen
should_use_val = opt.validation
dataset = opt.dataset
dataset_loader = getattr(gan_datasets, 'load_{}'.format(dataset))
fix_noise_dir = 'saved/fixednoise'
makedirs(fix_noise_dir)

CTX = mx.gpu() if opt.cuda else mx.cpu()
logger.info('Will use {}'.format(CTX))

# %% define dataloader
logger.info("Prepare data")
# noinspection PyTypeChecker
tfs_train = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(imageSize, imageSize), interpolation=2),
    # gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.RandomSaturation(0.001),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# noinspection PyTypeChecker
tfs_val = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(size=(imageSize, imageSize), interpolation=2),
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
generator = began.decoder(nz, imageSize, channel=3, nf=256)
discriminator = began.autoencoder(nz, imageSize, channel=3, nf=256)
generator.initialize(ctx=CTX)
discriminator.initialize(ctx=CTX)

print('Generator:')
generator.summary(mx.nd.random_normal(shape=(batch_size, nz), ctx=CTX))
print('\nDiscriminator:')
discriminator.summary(mx.nd.random_normal(shape=(batch_size, 3, imageSize, imageSize), ctx=CTX))

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
    history_labels = ['gloss', 'dloss', 'm']
history = TrainingHistory(labels=history_labels)

epsilon = 1e-10
k = epsilon
kLambda = 0.001
gamma = 0.5
l1loss = gluon.loss.L1Loss()
# scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[100, 150, 170, 200, 300, 310, 320], factor=0.5, base_lr=lr)
trainer_gen = gluon.Trainer(generator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr,
})
trainer_dis = gluon.Trainer(discriminator.collect_params(), optimizer='adam', optimizer_params={
    'learning_rate': lr
})

fix_noise_name = os.path.join(fix_noise_dir, '{}_{}'.format(nz, batch_size))
if os.path.exists(fix_noise_name):
    fix_noise = mx.nd.load(fix_noise_name)[0]
else:
    fix_noise = mx.nd.random_uniform(-1, 1, shape=(batch_size, nz))
    mx.nd.save(fix_noise_name, fix_noise)

# %% begin training
logger.info("Begin training")
dis_update_time = 0
gen_update_time = 0
iter4G = 100

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
        data = data.as_in_context(CTX)
        zD = mx.nd.random_uniform(-1, 1, shape=(batch_size, nz))
        zG = mx.nd.random_uniform(-1, 1, shape=(batch_size * 2, nz))

        ########################################
        # begin training discriminator
        ########################################
        dis_update_time += 1
        with autograd.record():
            # train with real image
            err2real = l1loss(discriminator(data), data)

            fake_img = generator(zD)
            weights = -epsilon * mx.nd.ones(batch_size)

            # train with fake image
            # detach the input, or its gradients will be computed
            err2fake = l1loss(discriminator(fake_img.detach()), fake_img, weights.reshape(-1, 1, 1, 1))

            err4dis = err2real + err2fake
            err4dis.backward()
        trainer_dis.step(batch_size)
        d_train_loss = err4dis.mean().asscalar()

        ########################################
        # begin training generator
        ########################################
        dis_update_time = 0
        gen_update_time += 1
        with autograd.record():
            target = generator(zG)
            fake = generator(zG)
            fake = discriminator(fake)
            err4gen = l1loss(fake, target)
            err4gen.backward()
        trainer_gen.step(batch_size)
        g_train_loss = err4gen.mean().asscalar()

        k = k + kLambda * (gamma * err2real.mean().asscalar() - err4gen.mean().asscalar())
        k = min(max(k, epsilon), 1)

        m_global = d_train_loss + np.abs(gamma * err2real.mean().asscalar() - g_train_loss)
        history.update([g_train_loss, d_train_loss, m_global])
        logger.info("Generator[{}], Discriminator[{}]".format(g_train_loss, d_train_loss))

        # make a prediction
        if gen_update_time % pred_per_gen == 0:
            pred_path = 'logs/pred-w1'
            makedirs(pred_path)

            pred = generator(fix_noise)
            vis.show_img(pred.transpose((0, 2, 3, 1)), save_path=pred_path, epoch=gen_update_time)

        # save checkpoint
        if should_save_checkpoint:
            if ep % save_per_epoch == 0:
                generator.save_parameters(os.path.join(save_dir, 'generator_{:04d}.params'.format(ep)))
                discriminator.save_parameters(os.path.join(save_dir, 'discriminator_{:04d}.params'.format(ep)))

        # clear current state
        g_train_loss = 0.0
        d_train_loss = 0.0
        g_iter_times = 0
        d_iter_times = 0

    # save history plot every epoch
    if ep % 10 == 0:
        history.plot(save_path='logs/histories-w1')

history.plot(save_path='logs/histories-w1')
