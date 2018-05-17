#
#   Darknet RegionLoss
#   Copyright EAVISE
#

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['RegionLoss']


class RegionLoss(nn.modules.loss._Loss):
    """ Computes region loss from darknet network output and target annotation.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        coord_scale (float): weight of bounding box coordinates
        noobject_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou between a predicted box and ground truth for them to be considered matching
        seen (int): How many images the network has already been trained on.
    """
    def __init__(self, num_classes, anchors, reduction=32, seen=0, coord_scale=1.0, noobject_scale=1.0, object_scale=5.0, class_scale=1.0, thresh=0.6):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction      # input_dim/output_dim
        self.seen = seen

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

    def forward(self, output, target, seen=None):
        """ Compute Region loss.

        Args:
            output (torch.autograd.Variable): Output from the network
            target (brambox.boxes.annotations.Annotation or torch.Tensor): Brambox annotations or tensor containing the annotation targets (see :class:`lightnet.data.BramboxToTensor`)
            seen (int, optional): How many images the network has already been trained on; Default **Add batch_size to previous seen value**

        Note:
            The example below only shows this function working with a target tensor. |br|
            This loss function also works with a list of brambox annotations as target and will work the same.
            The added benefit of using brambox annotations is that this function will then also look at the ``ignore`` flag of the annotations
            and ignore detections that match with it. This allows you to have annotations that will not influence the loss in any way,
            as opposed to having them removed and counting them as false detections.

        Example:
            >>> _ = torch.random.manual_seed(0)
            >>> network = ln.models.Yolo(num_classes=2, conf_thresh=4e-2)
            >>> region_loss = ln.network.loss.RegionLoss(network.num_classes, network.anchors)
            >>> Win, Hin = 96, 96
            >>> Wout, Hout = 1, 1
            >>> # true boxes for each item in the batch
            >>> # each box encodes class, x_center, y_center, width, and height
            >>> # coordinates are normalized in the range 0 to 1
            >>> # items in each batch are padded with dummy boxes with class_id=-1
            >>> target = torch.FloatTensor([
            ...     # boxes for batch item 1
            ...     [[0, 0.50, 0.50, 1.00, 1.00],
            ...      [1, 0.32, 0.42, 0.22, 0.12]],
            ...     # boxes for batch item 2 (it has no objects, note the pad!)
            ...     [[-1, 0, 0, 0, 0],
            ...      [-1, 0, 0, 0, 0]],
            ... ])
            >>> im_data = torch.autograd.Variable(torch.randn(len(target), 3, Hin, Win))
            >>> output = network._forward(im_data)
            >>> loss = float(region_loss(output, target))
            >>> print('loss = {loss:.2f}'.format(loss=loss))
            loss = 20.43
        """
        # Parameters
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cuda = output.is_cuda
        if isinstance(target, Variable):
            target = target.data
        if seen is not None:
            self.seen = seen
        else:
            self.seen += nB

        # Get x,y,w,h,conf,cls
        output = output.view(nB, nA, -1, nH*nW)
        coord = torch.zeros_like(output[:, :, :4])
        coord[:, :, :2] = output[:, :, :2].sigmoid()    # tx,ty
        coord[:, :, 2:4] = output[:, :, 2:4]            # tw,th
        conf = output[:, :, 4].sigmoid()
        if nC > 1:
            cls = output[:, :, 5:].contiguous().view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(-1, nC)

        # Create prediction boxes
        pred_boxes = torch.FloatTensor(nB*nA*nH*nW, 4)
        lin_x = torch.linspace(0, nW-1, nW).repeat(nH, 1).view(nH*nW)
        lin_y = torch.linspace(0, nH-1, nH).repeat(nW, 1).t().contiguous().view(nH*nW)
        anchor_w = self.anchors[:, 0].contiguous().view(nA, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(nA, 1)
        if cuda:
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        if torch.__version__.startswith('0.3'):
            pred_boxes[:, 0] = (coord[:, :, 0].data + lin_x).view(-1)
            pred_boxes[:, 1] = (coord[:, :, 1].data + lin_y).view(-1)
            pred_boxes[:, 2] = (coord[:, :, 2].data.exp() * anchor_w).view(-1)
            pred_boxes[:, 3] = (coord[:, :, 3].data.exp() * anchor_h).view(-1)
        else:
            pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
            pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
            pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
            pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()

        # Get target values
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target, nH, nW)
        coord_mask = coord_mask.expand_as(tcoord)
        if nC > 1:
            tcls = tcls[cls_mask].view(-1).long()
            cls_mask = cls_mask.view(-1, 1).repeat(1, nC)

        if cuda:
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            if nC > 1:
                tcls = tcls.cuda()
                cls_mask = cls_mask.cuda()

        conf_mask = conf_mask.sqrt()
        if torch.__version__.startswith('0.3'):
            tcoord = Variable(tcoord, requires_grad=False)
            tconf = Variable(tconf, requires_grad=False)
            coord_mask = Variable(coord_mask, requires_grad=False)
            conf_mask = Variable(conf_mask, requires_grad=False)
            if nC > 1:
                tcls = Variable(tcls, requires_grad=False)
                cls_mask = Variable(cls_mask, requires_grad=False)

        if nC > 1:
            cls = cls[cls_mask].view(-1, nC)

        # Compute losses
        mse = nn.MSELoss(size_average=False)
        self.loss_coord = self.coord_scale * mse(coord*coord_mask, tcoord*coord_mask) / nB
        self.loss_conf = mse(conf*conf_mask, tconf*conf_mask) / nB
        if nC > 1:
            self.loss_cls = self.class_scale * 2 * nn.CrossEntropyLoss(size_average=False)(cls, tcls) / nB
            self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
        else:
            self.loss_cls = None
            self.loss_tot = self.loss_coord + self.loss_conf

        return self.loss_tot

    def build_targets(self, pred_boxes, ground_truth, nH, nW):
        """ Compare prediction boxes and targets, convert targets to network output tensors """
        if torch.is_tensor(ground_truth):
            return self.__build_targets_tensor(pred_boxes, ground_truth, nH, nW)
        else:
            return self.__build_targets_brambox(pred_boxes, ground_truth, nH, nW)

    def __build_targets_tensor(self, pred_boxes, ground_truth, nH, nW):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        nB = ground_truth.size(0)
        nT = ground_truth.size(1)
        nA = self.num_anchors
        nAnchors = nA*nH*nW
        nPixels = nH*nW

        # Tensors
        if torch.__version__.startswith('0.3'):
            conf_mask = torch.ones(nB, nA, nH*nW) * self.noobject_scale
            coord_mask = torch.zeros(nB, nA, 1, nH*nW)
            cls_mask = torch.zeros(nB, nA, nH*nW).byte()
            tcoord = torch.zeros(nB, nA, 4, nH*nW)
            tconf = torch.zeros(nB, nA, nH*nW)
            tcls = torch.zeros(nB, nA, nH*nW)
        else:
            conf_mask = torch.ones(nB, nA, nH*nW, requires_grad=False) * self.noobject_scale
            coord_mask = torch.zeros(nB, nA, 1, nH*nW, requires_grad=False)
            cls_mask = torch.zeros(nB, nA, nH*nW, requires_grad=False).byte()
            tcoord = torch.zeros(nB, nA, 4, nH*nW, requires_grad=False)
            tconf = torch.zeros(nB, nA, nH*nW, requires_grad=False)
            tcls = torch.zeros(nB, nA, nH*nW, requires_grad=False)

        if self.seen < 12800:
            coord_mask.fill_(1)
            if self.anchor_step == 4:
                tcoord[:, :, 0] = self.anchors[:, 2].contiguous().view(1, nA, 1, 1).repeat(nB, 1, 1, nH*nW)
                tcoord[:, :, 1] = self.anchors[:, 3].contiguous().view(1, nA, 1, 1).repeat(nB, 1, 1, nH*nW)
            else:
                tcoord[:, :, 0].fill_(0.5)
                tcoord[:, :, 1].fill_(0.5)

        for b in range(nB):
            gt = ground_truth[b][(ground_truth[b, :, 0] >= 0)[:, None].expand_as(ground_truth[b])].view(-1, 5)
            if gt.dim() == 0 or gt.size(0) == 0:    # No gt for this image
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)

            gt = gt[:, 1:]
            gt[:, ::2] *= nW
            gt[:, 1::2] *= nH

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each gt
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each gt
            for i in range(gt.size(0)):
                gi = min(nW-1, max(0, int(gt[i, 0])))
                gj = min(nH-1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n*nPixels+gj*nW+gi]

                coord_mask[b][best_n][0][gj*nW+gi] = 1
                cls_mask[b][best_n][gj*nW+gi] = 1
                conf_mask[b][best_n][gj*nW+gi] = self.object_scale
                tcoord[b][best_n][0][gj*nW+gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj*nW+gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj*nW+gi] = math.log(gt[i, 2]/self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj*nW+gi] = math.log(gt[i, 3]/self.anchors[best_n, 1])
                tconf[b][best_n][gj*nW+gi] = iou
                tcls[b][best_n][gj*nW+gi] = ground_truth[b, i, 0]

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def __build_targets_brambox(self, pred_boxes, ground_truth, nH, nW):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        nB = len(ground_truth)
        nA = self.num_anchors
        nAnchors = nA*nH*nW
        nPixels = nH*nW

        # Tensors
        if torch.__version__.startswith('0.3'):
            conf_mask = torch.ones(nB, nA, nH*nW) * self.noobject_scale
            coord_mask = torch.zeros(nB, nA, 1, nH*nW)
            cls_mask = torch.zeros(nB, nA, nH*nW).byte()
            tcoord = torch.zeros(nB, nA, 4, nH*nW)
            tconf = torch.zeros(nB, nA, nH*nW)
            tcls = torch.zeros(nB, nA, nH*nW)
        else:
            conf_mask = torch.ones(nB, nA, nH*nW, requires_grad=False) * self.noobject_scale
            coord_mask = torch.zeros(nB, nA, 1, nH*nW, requires_grad=False)
            cls_mask = torch.zeros(nB, nA, nH*nW, requires_grad=False).byte()
            tcoord = torch.zeros(nB, nA, 4, nH*nW, requires_grad=False)
            tconf = torch.zeros(nB, nA, nH*nW, requires_grad=False)
            tcls = torch.zeros(nB, nA, nH*nW, requires_grad=False)

        if self.seen < 12800:
            coord_mask.fill_(1)
            if self.anchor_step == 4:
                tcoord[:, :, 0] = self.anchors[:, 2].contiguous().view(1, nA, 1, 1).repeat(nB, 1, 1, nH*nW)
                tcoord[:, :, 1] = self.anchors[:, 3].contiguous().view(1, nA, 1, 1).repeat(nB, 1, 1, nH*nW)
            else:
                tcoord[:, :, 0].fill_(0.5)
                tcoord[:, :, 1].fill_(0.5)

        for b in range(nB):
            if len(ground_truth[b]) == 0:   # No gt for this image
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
            gt = torch.zeros(len(ground_truth[b]), 4)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno.x_top_left + anno.width/2) / self.reduction
                gt[i, 1] = (anno.y_top_left + anno.height/2) / self.reduction
                gt[i, 2] = anno.width / self.reduction
                gt[i, 3] = anno.height / self.reduction

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each gt
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each gt
            for i, anno in enumerate(ground_truth[b]):
                gi = min(nW-1, max(0, int(gt[i, 0])))
                gj = min(nH-1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n*nPixels+gj*nW+gi]

                if anno.ignore:
                    conf_mask[b][best_n][gj*nW+gi] = 0
                else:
                    coord_mask[b][best_n][0][gj*nW+gi] = 1
                    cls_mask[b][best_n][gj*nW+gi] = 1
                    conf_mask[b][best_n][gj*nW+gi] = self.object_scale
                    tcoord[b][best_n][0][gj*nW+gi] = gt[i, 0] - gi
                    tcoord[b][best_n][1][gj*nW+gi] = gt[i, 1] - gj
                    tcoord[b][best_n][2][gj*nW+gi] = math.log(gt[i, 2]/self.anchors[best_n, 0])
                    tcoord[b][best_n][3][gj*nW+gi] = math.log(gt[i, 3]/self.anchors[best_n, 1])
                    tconf[b][best_n][gj*nW+gi] = iou
                    tcls[b][best_n][gj*nW+gi] = anno.class_id

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1_len = boxes1.size(0)
    b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions
