# %% import libs
import os
import argparse
import logging as logger
import mxnet as mx
import tqdm
from functools import partial
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.data.vision.datasets import ImageRecordDataset
from gluoncv.utils import makedirs

import datasets as gan_datasets
import models
from utils import vis, get_cpus, TrainingHistory, make_noise
from models.WasserStein import clip_dis
from models.WasserStein import weight_init

mx.random.seed(5)
logger.basicConfig(level=logger.INFO, filename='logs/train_loss-w1.log')

arg = argparse.ArgumentParser(description="training parameters")
arg.add_argument('--lr', type=float, default=0.00005, help='learning rate')
arg.add_argument('--nz', type=int, default=256, help='z vector dimension')
arg.add_argument('--imageSize', type=int, default=256, help='image size')
arg.add_argument('--extra_layers', type=int, default=0, help='extra layers for d & g')
arg.add_argument('--ngf', type=int, default=64)
arg.add_argument('--ndf', type=int, default=64)
arg.add_argument('--batch', type=int, default=8, help='batch size')
arg.add_argument('--epoch', type=int, default=5000000, help='training epochs')
arg.add_argument('--continue', type=bool, default=True, help='should continue with last checkpoint')
arg.add_argument('--save_checkpoint', type=bool, default=True, help='whether save checkpoint')
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
ngf = opt.ngf
ndf = opt.ndf
n_extra = opt.extra_layers
imageSize = opt.imageSize
should_save_checkpoint = opt.save_checkpoint
save_per_epoch = opt.save_per_epoch
save_dir = opt.save_dir
pred_per_gen = opt.pred_per_gen
should_use_val = opt.validation
dataset = opt.dataset
dataset_loader = getattr(gan_datasets, 'load_{}'.format(dataset))

CTX = mx.gpu() if opt.cuda else mx.cpu()
logger.info('Will use {}'.format(CTX))
make_noise = partial(make_noise, batch_size, nz, CTX)

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
generator = models.make_generic_gen(imageSize, nz=nz, nc=3, ngf=32, n_extra_layers=1)
discriminator = models.make_generic_dis(imageSize, nc=3, ndf=32, n_extra_layers=1)
generator.initialize(init=mx.init.Xavier(factor_type='in', magnitude=0.01), ctx=CTX)
discriminator.initialize(init=mx.init.Xavier(factor_type='in', magnitude=0.01), ctx=CTX)

print('Generator:')
generator.summary(mx.nd.random_normal(shape=(batch_size, nz, 1, 1), ctx=CTX))
print('\nDiscriminator:')
discriminator.summary(mx.nd.random_normal(shape=(batch_size, 3, imageSize, imageSize), ctx=CTX))

# reinitialize the weights
weight_init(discriminator)
weight_init(generator)

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
# scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[100, 150, 170, 200, 300, 310, 320], factor=0.5, base_lr=lr)
trainer_gen = gluon.Trainer(generator.collect_params(), optimizer='rmsprop', optimizer_params={
    'learning_rate': lr,
    'epsilon': 1e-13
})
# you can use weight clip in the training scope
trainer_dis = gluon.Trainer(discriminator.collect_params(), optimizer='rmsprop', optimizer_params={
    'learning_rate': lr,
    'epsilon': 1e-11,
    # 'clip_weights': 0.01
})

fix_noise_name = 'saved/fixednoise/{}_{}'.format(nz, batch_size)
if os.path.exists(fix_noise_name):
    fix_noise = mx.nd.load(fix_noise_name)[0]
else:
    fix_noise = make_noise()
    mx.nd.save(fix_noise_name, fix_noise)

# %% begin training
logger.info("Begin training")
dis_update_time = 0
gen_update_time = 0
iter4G = 100

g_train_loss = 0.0
d_train_loss = 0.0
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

        # use clip operation in optimizer
        # clip the discriminator
        clip_dis(discriminator, 0.01)

        ########################################
        # begin training discriminator
        ########################################
        if dis_update_time < iter4G:
            dis_update_time += 1
            with autograd.record():
                # train with real image
                err2real = discriminator(data).mean(axis=0).reshape(1)

                fake_img = generator(make_noise())

                # train with fake image
                # detach the input, or its gradients will be computed
                err2fake = discriminator(fake_img.detach()).mean(axis=0).reshape(1)

                err4dis = err2real - err2fake
                err4dis.backward()
            trainer_dis.step(1)
            d_train_loss = err4dis.asscalar()

        ########################################
        # begin training generator
        ########################################
        if dis_update_time == iter4G:
            dis_update_time = 0
            gen_update_time += 1
            with autograd.record():
                fake_img = generator(make_noise())
                err4gen = discriminator(fake_img).mean(axis=0).reshape(1)
                err4gen.backward()
            trainer_gen.step(1)
            g_train_loss = err4gen.asscalar()

            history.update([g_train_loss, d_train_loss])
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

            # save history plot every epoch
            history.plot(save_path='logs/histories-w1')

            # clear current state
            g_train_loss = 0.0
            d_train_loss = 0.0
            g_iter_times = 0
            d_iter_times = 0

            # D:G training schedule
            if gen_update_time > 25:
                iter4G = 5

history.plot(save_path='logs/histories-w1')
