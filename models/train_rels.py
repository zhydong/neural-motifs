"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""
import _init_paths

import os
import time
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ModelConfig, BOX_SCALE, IM_SCALE
from dataloaders.visual_genome import VGDataLoader, VG
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from lib.object_detector import ObjectDetector

from IPython import embed

print('loading configuration')
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
elif conf.model == 'fcknet_v1':
    from lib.my_model_v1 import FckModel as RelModel
elif conf.model == 'fcknet_v2':
    from lib.my_model_v2 import FckModel as RelModel
elif conf.model == 'fcknet_v3':
    from lib.my_model_v3 import FckModel as RelModel
else:
    raise ValueError()

print('loading dataloader')
train, val, _ = VG.splits(
    num_val_im=conf.val_size,
    filter_duplicate_rels=True,
    use_proposals=conf.use_proposals,
    filter_non_overlap=conf.mode == 'sgdet'
)
train_loader, val_loader = VGDataLoader.splits(
    train,
    val,
    mode='rel',
    batch_size=conf.batch_size,
    num_workers=conf.num_workers,
    num_gpus=conf.num_gpus
)

# embed(header='fuck in train_rel.py')
print('loading model')
detector = RelModel(
    classes=train.ind_to_classes,
    rel_classes=train.ind_to_predicates,
    mode=conf.mode,
    num_gpus=conf.num_gpus,
    use_vision=True,
    require_overlap_det=True,
    use_resnet=conf.use_resnet,
    order=conf.order,
    nl_edge=conf.nl_edge,
    nl_obj=conf.nl_obj,
    hidden_dim=conf.hidden_dim,
    use_proposals=conf.use_proposals,
    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
    obj_dim=conf.obj_dim,
    pooling_dim=conf.pooling_dim,
    rec_dropout=conf.rec_dropout,
    use_bias=conf.use_bias,
    use_tanh=conf.use_tanh,
    limit_vision=conf.limit_vision
)

detector.cuda()

if conf.use_tf:
    from tensorboardX import SummaryWriter
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_dir = osp.join('log', conf.model, 'train_%s' % time_str)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    tf_writer = SummaryWriter(
        log_dir=log_dir,
        comment='FckNet'
    )

print('freezing parameter.....')
# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n, p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n, p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler


ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])

    if conf.model.split('_')[0] == 'fcknet':
        detector.roi_fmap[0][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        #detector.roi_fmap[0][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap[0][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        #detector.roi_fmap[0][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
    else:
        detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
        detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])



def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        res = train_batch(batch, b)
        if res is None:
            continue
        else:
            tr.append(res)

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            #try:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            #except:
            #embed(header='concat')
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)

            if conf.use_tf:
                info = {
                    'loss': mn['total'],
                    'class_loss': mn['class_loss'],
                    'rel_loss': mn['rel_loss'],
                    'rel_pn_loss': mn['rel_pn_loss'],
                    'rel_trim_ratio': mn['trim_pos'] / mn['trim_total'],
                    'rel_sample_ratio': mn['sample_pos'] / (mn['sample_pos'] + mn['sample_neg']),
                }
                tf_writer.add_scalars(
                    main_tag='.',
                    tag_scalar_dict=info,
                    global_step=b + epoch_num * len(train_loader)  # TOBE CHECK, batch num in an epoch
                )

            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(batch, bi):
    """
    Args:
        batch: contains:
            imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
            all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
            all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
            im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

            num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
            train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
            gt_boxes: [num_gt, 4] GT boxes over the batch.
            gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        bi: batch index, integer
    Returns:
        result: pd.Series, result dict
    """
    result = detector[batch]
    if result is None:
        print('Error! No Pos Relation', bi)
        return

    losses = dict()

    class_loss = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    losses['class_loss'] = class_loss.data[0]

    rel_loss = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])
    losses['rel_loss'] = rel_loss.data[0]

    loss = class_loss + rel_loss
    if conf.model.split('_')[0] == 'fcknet':
        rel_pn_loss = F.cross_entropy(result.rel_pn_dists, result.rel_pn_labels)
        losses['rel_pn_loss'] = rel_pn_loss.data[0]
        loss += rel_pn_loss

    # TODO
    if conf.model == 'fcknet_v3':
        rel_mem_loss = F.cross_entropy(result.rel_mem_dists, result.rel_labels[:, -1])
        losses['mem_loss'] = rel_mem_loss.data[0]
        loss += rel_mem_loss
    if bi % conf.print_interval == 0 and bi >= conf.print_interval:
        if conf.model.split('_')[0] == 'fcknet':
            print(
                'rel_pn_loss: %.4f, cls_loss: %.4f, rel_loss: %.4f' %
                (losses['rel_pn_loss'], losses['class_loss'], losses['rel_loss'])
            )
            if conf.model == 'fcknet_v3':
                print(
                    'rel_mem_loss: %.4f' % losses['mem_loss']
                )

        else:
            print(
                'cls_loss: %.4f, rel_loss: %.4f' %
                (losses['class_loss'], losses['rel_loss'])
            )

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=bi % (conf.print_interval*10) == 0, clip=True
    )

    losses['total'] = loss.data[0]
    losses['trim_pos'] = result.rel_trim_pos[0]
    losses['trim_total'] = result.rel_trim_total[0]
    losses['sample_pos'] = result.rel_sample_pos[0]
    losses['sample_neg'] = result.rel_sample_neg[0]
    losses['relpn_recall'] = result.rel_pn_recall[0]
    optimizer.step()
    res = pd.Series({x: y for x, y in losses.items()})
    return res


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats()
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if det_res is None:
        return
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0), 'Object class not right'
        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


print("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)
mAp = val_epoch()
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
            #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    mAp = val_epoch()
    scheduler.step(mAp)
    if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
        print("exiting training early", flush=True)
        break

if conf.use_tf:
    tf_writer.close()
