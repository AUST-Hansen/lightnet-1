#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Test Yolo on Pascal VOC
#

import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import visdom
import hyperdash
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln
import time

log = logging.getLogger('lightnet.test')
ln.logger.setLogFile('best_pr.log', filemode='a')           # Enable logging of test logs (By appending, multiple runs will keep writing to same file, allowing to search the best)
#ln.logger.setConsoleLevel(logging.NOTSET)                  # Enable debug prints in terminal
#ln.logger.setConsoleColor(False)                           # Disable colored terminal output

# Parameters
WORKERS = 20
PIN_MEM = True
ROOT = 'data'
TESTFILE = '{ROOT}/test.pkl'.format(ROOT=ROOT)

VISDOM_PORT = 8097

LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CLASSES = len(LABELS)

NETWORK_SIZE = (416, 416)
CONF_THRESH = 0.001
NMS_THRESH = 0.4

BATCH = 64
MINI_BATCH = 8


class CustomDataset(ln.data.BramboxData):
    def __init__(self, anno, network):
        def identify(img_id):
            return '{ROOT}/VOCdevkit/{img_id}'.format(ROOT=ROOT, img_id=img_id)

        lb  = ln.data.transform.Letterbox(NETWORK_SIZE)
        it  = tf.ToTensor()
        img_tf = ln.data.transform.Compose([lb, it])
        anno_tf = ln.data.transform.Compose([lb])

        super(CustomDataset, self).__init__('anno_pickle', anno, NETWORK_SIZE, LABELS, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        img, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return img, anno


def test(arguments):
    log.debug('Creating network')
    net = ln.models.Yolo(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
    net = torch.nn.DataParallel(net)
    net.postprocess.append(ln.data.transform.TensorToBrambox(NETWORK_SIZE, LABELS))

    net.eval()
    if arguments.cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        CustomDataset(TESTFILE, net),
        batch_size=MINI_BATCH,
        shuffle=False,
        drop_last=False,
        num_workers=WORKERS if arguments.cuda else 0,
        pin_memory=PIN_MEM if arguments.cuda else False,
        collate_fn=ln.data.list_collate,
    )

    if arguments.visdom:
        log.debug('Creating visdom visualisation wrappers')
        vis = visdom.Visdom(port=VISDOM_PORT)
        visdom_plot_pr = ln.engine.VisdomLinePlotter(vis, 'pr', opts=dict(xlabel='Recall', ylabel='Precision', title='Precision Recall', xtickmin=0, xtickmax=1, ytickmin=0, ytickmax=1, showlegend=True))

    if arguments.hyperdash:
        log.debug('Creating hyperdash visualisation wrappers')
        hd = hyperdash.Experiment('YOLOv2 Pascal VOC Test')
        hyperdash_plot_pr = ln.engine.HyperdashLinePlotter(hd)

    log.debug('Running network')
    tot_loss = []
    coord_loss = []
    conf_loss = []
    cls_loss = []
    anno, det = {}, {}

    for idx, (data, box) in enumerate(tqdm(loader, total=len(loader))):
        if arguments.cuda:
            data = data.cuda()

        if torch.__version__.startswith('0.3'):
            data = torch.autograd.Variable(data, volatile=True)
            output, loss = net(data, box)
        else:
            with torch.no_grad():
                output, loss = net(data, box)

        if torch.__version__.startswith('0.3'):
            tot_loss.append(net.loss.loss_tot.data[0] * len(box))
            coord_loss.append(net.loss.loss_coord.data[0] * len(box))
            conf_loss.append(net.loss.loss_conf.data[0] * len(box))
            if net.loss.loss_cls is not None:
                cls_loss.append(net.loss.loss_cls.data[0] * len(box))
        else:
            tot_loss.append(net.loss.loss_tot.item() * len(box))
            coord_loss.append(net.loss.loss_coord.item() * len(box))
            conf_loss.append(net.loss.loss_conf.item() * len(box))
            if net.loss.loss_cls is not None:
                cls_loss.append(net.loss.loss_cls.item() * len(box))

        key_val = len(anno)
        anno.update({loader.dataset.keys[key_val + k]: v for k, v in enumerate(box)})
        det.update({loader.dataset.keys[key_val + k]: v for k, v in enumerate(output)})

    log.debug('Computing statistics')

    pr = bbb.pr(det, anno)
    m_ap = round(bbb.ap(*pr) * 100, 2)
    tot = round(sum(tot_loss) / len(anno), 5)
    coord = round(sum(coord_loss) / len(anno), 2)
    conf = round(sum(conf_loss) / len(anno), 2)
    if len(cls_loss) > 0:
        cls = round(sum(cls_loss) / len(anno), 2)
        log.test('{seen} mAP:{m_ap}% Loss:{tot} (Coord:{coord} Conf:{conf} Cls:{cls})'.format(seen=net.seen // BATCH, m_ap=m_ap, tot=tot, coord=coord, conf=conf, cls=cls))
    else:
        log.test('{seen} mAP:{m_ap}% Loss:{tot} (Coord:{coord} Conf:{conf})'.format(seen=net.seen // BATCH, m_ap=m_ap, tot=tot, coord=coord, conf=conf))

    name = 'mAP: {m_ap}%'.format(m_ap=m_ap)
    if arguments.visdom:
        visdom_plot_pr(np.array(pr[0]), np.array(pr[1]), name=name)

    if arguments.hyperdash:
        now = time.time()
        re_seen = None
        for index, (re_, pr_) in enumerate(sorted(zip(pr[1], pr[0]))):
            re_ = round(re_, 2)
            if re_ != re_seen:
                re_seen = re_
                re_ = int(re_ * 100.0)
                hyperdash_plot_pr(name, pr_, now + re_, log=False)

    if arguments.save_det is not None:
        # Note: These detection boxes are the coordinates for the letterboxed images,
        #       you need ln.data.ReverseLetterbox to have the right ones.
        #       Alternatively, you can save the letterboxed annotations, and use those for statistics later on!
        bbb.generate('det_pickle', det, Path(arguments.save_det).with_suffix('.pkl'))
        #bbb.generate('anno_pickle', det, Path('anno-letterboxed_'+arguments.save_det).with_suffix('.pkl'))

    if arguments.hyperdash:
        hyperdash_plot_pr.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    parser.add_argument('--hyperdash', '--hd', action='store_true', help='Visualize training data with hyperdash')
    parser.add_argument('-s', '--save_det', help='Save detections as a brambox pickle file', default=None)
    args = parser.parse_args()

    # Parse arguments
    if args.cuda:
        if not torch.cuda.is_available():
            log.error('CUDA not available')
            args.cuda = False
        else:
            log.debug('CUDA enabled')

    # Test
    test(args)
