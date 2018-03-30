#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Train the lightnet yolo network using the lightnet engine
#            This example script uses darknet type annotations
#

import os
import argparse
import logging
from statistics import mean
import visdom
import hyperdash
import numpy as np
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln
import time
from tqdm import tqdm

log = logging.getLogger('lightnet.train')
ln.logger.setLogFile('train.log', filemode='w')             # Enable logging of TRAIN and TEST logs
#ln.logger.setConsoleLevel(logging.DEBUG)                   # Enable debug log messages in terminal
#ln.logger.setConsoleColor(False)                           # Disable colored terminal log messages

# Parameters
WORKERS = 4
PIN_MEM = True
ROOT = 'data'
TRAINFILE = '{ROOT}/train.pkl'.format(ROOT=ROOT)
TESTFILE  = '{ROOT}/valid.pkl'.format(ROOT=ROOT)
VISDOM_PORT = 8097

LABELS = ['shark_hammerhead']
CLASSES = len(LABELS)

NETWORK_SIZE = (416, 416)
CONF_THRESH = 0.01
NMS_THRESH = 0.80

BATCH = 16
MINI_BATCH = 16
MAX_BATCHES = 64000

JITTER = 0.2
FLIP = 0.5
HUE = 0.1
SAT = 1.5
VAL = 1.5

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
DECAY = 0.0005
LR_STEPS = [  100,  25000,   45000]
LR_RATES = [0.001, 0.0001, 0.00001]

BACKUP = 500
BP_STEPS = [5000, 50000]
BP_RATES = [1000, 5000]

TEST = 100
TS_STEPS = []
TS_RATES = []

RESIZE = 10
RS_STEPS = []
RS_RATES = []


class VOCDataset(ln.data.BramboxData):
    def __init__(self, anno):
        def identify(img_id):
            return '{ROOT}/VOCdevkit/{img_id}'.format(ROOT=ROOT, img_id=img_id)

        lb  = ln.data.Letterbox(dataset=self)
        rf  = ln.data.RandomFlip(FLIP)
        rc  = ln.data.RandomCrop(JITTER, True, 0.1)
        hsv = ln.data.HSVShift(HUE, SAT, VAL)
        it  = tf.ToTensor()
        img_tf = tf.Compose([hsv, rc, rf, lb, it])
        anno_tf = tf.Compose([rc, rf, lb])

        super(VOCDataset, self).__init__('anno_pickle', anno, NETWORK_SIZE, LABELS, identify, img_tf, anno_tf)


class TrainingEngine(ln.engine.Engine):
    """This is a custom engine for this training cycle."""

    batch_size = BATCH
    mini_batch_size = MINI_BATCH
    max_batches = MAX_BATCHES

    def __init__(self, arguments, **kwargs):
        self.cuda = arguments.cuda
        self.backup_folder = arguments.backup
        self.enable_testing = arguments.test
        self.visdom = arguments.visdom
        self.hyperdash = arguments.hyperdash

        log.debug('Creating network')
        net = ln.models.Yolo(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
        net.postprocess = tf.Compose([
            net.postprocess,
            ln.data.TensorToBrambox(NETWORK_SIZE, LABELS),
        ])
        if self.cuda:
            net.cuda()

        log.debug('Creating optimizer')
        optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE / BATCH, momentum=MOMENTUM, dampening=0, weight_decay=DECAY * BATCH)

        log.debug('Creating datasets')
        data = ln.data.DataLoader(
            VOCDataset(TRAINFILE),
            batch_size=MINI_BATCH,
            shuffle=True,
            drop_last=True,
            num_workers=WORKERS if self.cuda else 0,
            pin_memory=PIN_MEM if self.cuda else False,
            collate_fn=ln.data.list_collate,
        )
        if self.enable_testing is not None:
            self.testloader = ln.data.DataLoader(
                VOCDataset(TESTFILE),
                batch_size=MINI_BATCH,
                shuffle=True,
                drop_last=True,
                num_workers=WORKERS if self.cuda else 0,
                pin_memory=PIN_MEM if self.cuda else False,
                collate_fn=ln.data.list_collate,
            )

        super(TrainingEngine, self).__init__(net, optim, data)

    def start(self):
        """Starting values."""
        if CLASSES > 1:
            legend = ['Total loss', 'Coordinate loss', 'Confidence loss', 'Class loss']
        else:
            legend = ['Total loss', 'Coordinate loss', 'Confidence loss']

        self.visdom_plot_train_loss = ln.engine.VisdomLinePlotter(
            self.visdom,
            'train_loss',
            opts=dict(
                title='Training Loss',
                xlabel='Batch',
                ylabel='Loss',
                showlegend=True,
                legend=legend,
            )
        )

        self.hyperdash_plot_train_loss = ln.engine.HyperdashLinePlotter(
            self.hyperdash,
            opts={
                'Batch Size': self.batch_size,
            }
        )

        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}
        self.add_rate('learning_rate', LR_STEPS, [lr / BATCH for lr in LR_RATES])
        self.add_rate('backup_rate', BP_STEPS, BP_RATES, BACKUP)
        self.add_rate('resize_rate', RS_STEPS, RS_RATES, RESIZE)

        if self.enable_testing is not None:
            self.visdom_plot_test_loss = ln.engine.VisdomLinePlotter(
                self.visdom,
                'test_loss',
                name='Total loss',
                opts=dict(
                    title='Testing Loss',
                    xlabel='Batch',
                    ylabel='Loss',
                    showlegend=True
                )
            )
            self.add_rate('test_rate', TS_STEPS, TS_RATES, TEST)

        self.dataloader.change_input_dim()

    def process_batch(self, data):
        data, target = data
        if self.cuda:
            data = data.cuda(async=PIN_MEM)
        data = torch.autograd.Variable(data, requires_grad=True)

        loss = self.network(data, target)
        loss.backward()

        self.train_loss['tot'].append(self.network.loss.loss_tot.data[0])
        self.train_loss['coord'].append(self.network.loss.loss_coord.data[0])
        self.train_loss['conf'].append(self.network.loss.loss_conf.data[0])
        if self.network.loss.loss_cls is not None:
            self.train_loss['cls'].append(self.network.loss.loss_cls.data[0])

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        tot = mean(self.train_loss['tot'])
        coord = mean(self.train_loss['coord'])
        conf = mean(self.train_loss['conf'])
        if CLASSES > 1:
            cls = mean(self.train_loss['cls'])
        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}

        if CLASSES > 1:
            self.visdom_plot_train_loss(np.array([[tot, coord, conf, cls]]), np.array([self.batch]))
            self.log('{self.batch} Loss:{tot:0.05f} (Coord:{coord:0.02f} Conf:{conf:0.02f} Cls:{cls:0.02f})'.format(
                self=self,
                tot=tot,
                coord=coord,
                conf=conf,
                cls=cls
            ))
        else:
            self.visdom_plot_train_loss(np.array([[tot, coord, conf]]), np.array([self.batch]))
            self.log('{self.batch} Loss:{tot:0.05f} (Coord:{coord:0.02f} Conf:{conf:0.02f})'.format(
                self=self,
                tot=tot,
                coord=coord,
                conf=conf
            ))

        self.hyperdash_plot_train_loss('Loss Total', tot, log=False)
        self.hyperdash_plot_train_loss('Loss Coordinate', coord, log=False)
        self.hyperdash_plot_train_loss('Loss Confidence', conf, log=False)
        self.hyperdash_plot_train_loss('Loss Class', cls, log=False)

        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_folder, 'weights_{self.batch}.pt'.format(self=self)))

        if self.batch % self.resize_rate == 0:
            self.dataloader.change_input_dim()

    def test(self):
        tot_loss = []
        anno, det = {}, {}

        for idx, (data, target) in enumerate(tqdm(self.testloader, total=len(self.testloader))):
            if self.cuda:
                data = data.cuda(async=PIN_MEM)

            data = torch.autograd.Variable(data, volatile=True)

            output, loss = self.network(data, target)

            tot_loss.append(loss.data[0] * len(target))
            key_val = len(anno)
            anno.update({key_val + k: v for k, v in enumerate(target)})
            det.update({key_val + k: v for k, v in enumerate(output)})

            if self.sigint:
                return

        pr = bbb.pr(det, anno)
        m_ap = bbb.ap(*pr)
        loss = round(sum(tot_loss) / len(anno), 5)
        self.log('\nLoss:{loss} mAP:{m_ap:0.02f}%'.format(loss=loss, m_ap=m_ap * 100.0))
        self.visdom_plot_test_loss(np.array([loss]), np.array([self.batch]))
        self.hyperdash_plot_train_loss('Loss Total (Test)', loss, log=False)
        self.hyperdash_plot_train_loss('mAP (Test)', m_ap, log=False)

    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_folder, 'backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_folder, 'final.pt'))
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('-b', '--backup', help='Backup folder', default='./backup')
    parser.add_argument('-t', '--test', action='store_true', help='Enable testing')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    parser.add_argument('--hyperdash', '--hd', action='store_true', help='Visualize training data with hyperdash')
    args = parser.parse_args()

    # Parse arguments
    if args.cuda:
        if not torch.cuda.is_available():
            log.error('CUDA not available')
            args.cuda = False
        else:
            log.debug('CUDA enabled')

    if args.visdom:
        args.visdom = visdom.Visdom(port=VISDOM_PORT)
    else:
        args.visdom = None

    if args.hyperdash:
        args.hyperdash = hyperdash.Experiment('YOLOv2 Train')
    else:
        args.hyperdash = None

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            log.warn('Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            raise ValueError('Backup path is not a folder')

    # Train
    eng = TrainingEngine(args)
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    print('\nDuration of {duration} batches: {batches} seconds [{seconds:0.03f} sec/batch]'.format(
        duration=b2 - b1,
        batches=t2 - t1,
        seconds=(t2 - t1) / (b2 - b1),
    ))

    if eng.hyperdash_plot_train_loss is not None:
        eng.hyperdash_plot_train_loss.close()
