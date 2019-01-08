"""
Add message passing
Add Memory
"""

import math
import time
from time import time as tt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM, RELEVANT_PER_IM, EDGES_PER_IM
from lib.fpn.nms.functions.nms import apply_nms
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import \
    transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.sparse_targets import FrequencyBias
from lib.get_dataset_counts import get_counts
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.lstm.mem_rnn import MemoryRNN

from IPython import embed


MODES = ('sgdet', 'sgcls', 'predcls')


class FckModel(nn.Module):
    """
    1 ) Proposal Relation
    2 ) fc layer predict
    3 ) memory embedding
    """
    def __init__(
            self,
            classes,
            rel_classes,
            mode='sgdet',
            num_gpus=1,
            use_vision=True,
            require_overlap_det=False,
            embed_dim=200,
            hidden_dim=256,
            obj_dim=2048,
            pooling_dim=4096,
            nl_obj=1,
            nl_edge=2,
            use_resnet=True,
            order='confidence',
            thresh=0.01,
            use_proposals=False,
            pass_in_obj_feats_to_decoder=True,
            pass_in_obj_feats_to_edge=True,
            rec_dropout=0.0,
            use_bias=True,
            use_tanh=True,
            limit_vision=True,
            spatial_dim=128,
            mp_iter_num=1,
            trim_graph=True
    ):
        """
        Args:
            mp_iter_num: integer, number of message passing iteration
            trim_graph: boolean, trim graph in rel pn
        """
        super(FckModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.pooling_dim = 2048 if use_resnet else 4096
        self.spatial_dim = spatial_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.mp_iter_num = mp_iter_num
        self.trim_graph = trim_graph

        classes_word_vec = obj_edge_vectors(self.classes, wv_dim=embed_dim)
        self.classes_word_embedding = nn.Embedding(self.num_classes, embed_dim)
        self.classes_word_embedding.weight.data = classes_word_vec.clone()
        self.classes_word_embedding.weight.requires_grad = False

        #fg_matrix, bg_matrix = get_counts()
        #rel_obj_distribution = fg_matrix / (fg_matrix.sum(2)[:, :, None] + 1e-5)
        #rel_obj_distribution = torch.FloatTensor(rel_obj_distribution)
        #rel_obj_distribution = rel_obj_distribution.view(-1, self.num_rels)
#
        #self.rel_obj_distribution = nn.Embedding(rel_obj_distribution.size(0), self.num_rels)
        ## (#obj_class * #obj_class, #rel_class)
        #self.rel_obj_distribution.weight.data = rel_obj_distribution

        if mode == 'sgdet':
            if use_proposals:
                obj_detector_mode = 'proposals'
            else:
                obj_detector_mode = 'refinerels'
        else:
            obj_detector_mode = 'gtbox'

        self.detector = ObjectDetector(
            classes=classes,
            mode=obj_detector_mode,
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )

        self.union_boxes = UnionBoxesAndFeats(
            pooling_size=self.pooling_size,
            stride=16,
            dim=1024 if use_resnet else 512,
            use_feats=False
        )
        self.spatial_fc = nn.Sequential(*[
            nn.Linear(4, spatial_dim),
            nn.BatchNorm1d(spatial_dim, momentum=BATCHNORM_MOMENTUM / 10.),
            nn.ReLU(inplace=True)
        ])
        self.word_fc = nn.Sequential(*[
            nn.Linear(2*embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=BATCHNORM_MOMENTUM / 10.),
            nn.ReLU(inplace=True)
        ])
        # union box feats
        feats_dim = obj_dim + spatial_dim + hidden_dim
        self.relpn_fc = nn.Linear(feats_dim, 2)
        self.relcnn_fc1 = nn.Sequential(*[
            nn.Linear(feats_dim, feats_dim),
            nn.ReLU(inplace=True)
        ])

        # v2 model---------
        self.box_mp_fc = nn.Sequential(*[
            nn.Linear(obj_dim, obj_dim),
        ])
        self.sub_rel_mp_fc = nn.Sequential(*[
            nn.Linear(feats_dim, obj_dim)
        ])

        self.obj_rel_mp_fc = nn.Sequential(*[
            nn.Linear(feats_dim, obj_dim),
        ])

        self.mp_atten_fc = nn.Sequential(*[
            nn.Linear(feats_dim + obj_dim, obj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obj_dim, 1)
        ])
        # v2 model----------

        self.cls_fc = nn.Linear(obj_dim, self.num_classes)

        self.relcnn_fc2 = nn.Linear(feats_dim, self.num_rels)

        # v3 model -----------

        self.mem_module = MemoryRNN(
            classes=classes,
            rel_classes=rel_classes,
            inputs_dim=feats_dim,
            hidden_dim=hidden_dim,
            recurrent_dropout_probability=.0
        )
        # v3 model -----------

        if use_resnet:
            # deprecate
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                load_vgg(
                    use_dropout=False,
                    use_relu=False,
                    use_linear=self.obj_dim == 4096,
                    pretrained=False,
                ).classifier,
                nn.Linear(self.pooling_dim, self.obj_dim)

            ]
            self.roi_fmap = nn.Sequential(*roi_fmap)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """Classify the features
        Args:
            features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
            rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
            pair_inds: inds to use when predicting
        Returns:
            score_pred: a [num_rois, num_classes] array
            box_pred: a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes.view(uboxes.size(0), -1))

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        """Get relation index
        Args:
            rel_labels: Variable
            im_inds: Variable
            box_priors: Variable
        """
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.contiguous().clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None].contiguous(), rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        Args:
            features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
            rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        Returns:
            features: [num_rois, #dim] array
        """
        roi_align_func = RoIAlignFunction(
            self.pooling_size,
            self.pooling_size,
            spatial_scale=1/16
        )
        feature_pool = roi_align_func(features, rois)
        return self.roi_fmap(feature_pool.view(rois.size(0), -1))

    def fuse_message(self, union_box_feats, boxes, box_classes, rel_inds):
        """Fuse union Appearance features, box spatial information and NLP word features together
        Args:
            union_box_feats: Variable
            boxes: Variable
            box_classes: Variable
            rel_inds: Variable
        Returns:
            box_pair_feats:
        """
        bboxes = Variable(center_size(boxes.data))
        sub_bboxes = bboxes[rel_inds[:, 1].contiguous()]
        obj_bboxes = bboxes[rel_inds[:, 2].contiguous()]

        obj_bboxes[:, :2] = obj_bboxes[:, :2].contiguous() - sub_bboxes[:, :2].contiguous()  # x-y
        obj_bboxes[:, 2:] = obj_bboxes[:, 2:].contiguous() / sub_bboxes[:, 2:].contiguous()  # w/h
        obj_bboxes[:, :2] /= sub_bboxes[:, 2:].contiguous()  # x-y/h
        obj_bboxes[:, 2:] = torch.log(obj_bboxes[:, 2:].contiguous())  # log(w/h)

        bbox_spatial_feats = self.spatial_fc(obj_bboxes)

        box_word = self.classes_word_embedding(box_classes)
        box_pair_word = torch.cat(
            (box_word[rel_inds[:, 1].contiguous()], box_word[rel_inds[:, 2].contiguous()]), 1
        )
        box_word_feats = self.word_fc(box_pair_word)

        # (NumOfRels, DIM=)
        box_pair_feats = torch.cat(
            (union_box_feats, bbox_spatial_feats, box_word_feats), 1
        )
        return box_pair_feats

    def message_passing(self, box_feats, rel_feats, edges):
        """Integrate box feats to each other
        update box feats by decending out-degree order, that is, the node with largest out-degree update first

        suppose node i and j are neighbours, and has connection i->j
        feature of i and j are fi and fj, feature of union box ij are fij

        fi = sigma(W1*fi + sum_neighbour(V1 * alpha * fij))
        fj = sigma(W2*fj + sum_neighbour(V2 * alpha * fij))

        alpha = attention(fi, fij)

        V1, V2, W1, W2 are parameters to be learned, sigma is acitvation function, alpha is attention
        Args:
            box_feats: Variable, box features with shape of (NumOfBoxes, FEAT_DIM)
            rel_feats: Variable, edge features with shape of (NumOfRels, REL_FEAT_DIM)
            edges: Variable, scene graph edges(pruned), with shape of (NumOfRels, 3)
                e.g. edges[0, :] = [1, 0, 5] means box 0 and box 5 in image 1 had an affair~
        Returns:
            box_feats: Variable, box features combining relation features
        """
        # embed(header='mp ')
        im_inds = edges[:, 0].contiguous()
        num_img = im_inds[-1] + 1
        # list of dict: record the number of boxes per image
        count_dic = [{} for _ in range(num_img)]
        for im_i, s, e in enumerate_by_image(im_inds):
            im_i_edges = edges[s:e, :].contiguous()
            for rel in im_i_edges:
                box0, box1 = rel[1:]
                count_dic[im_i][box0] = 1 + count_dic[im_i].get(box0, 0)

        # list of Variable
        box_nodes_feats = list()
        for box_feat in box_feats:
            box_nodes_feats.append(box_feat)#.clone())

        for im_i, s, e in enumerate_by_image(im_inds):
            im_i_edges = edges[s:e, :].contiguous()
            im_i_rel_feats = rel_feats[s:e, :].contiguous()
            for box_id, v in \
                    sorted(
                        count_dic[im_i].items(),
                        key=lambda kv: kv[1],
                        reverse=True
                    ):
                # update passing message
                # subject message from rel feats
                choose_sub_edges_ind = torch.nonzero(im_i_edges[:, 1].contiguous() == box_id).squeeze()
                choose_sub_edges = im_i_edges[choose_sub_edges_ind]
                choose_sub_rel_feats = im_i_rel_feats[choose_sub_edges_ind]
                box_id_feats = box_nodes_feats[box_id]

                # attention on subject reltions
                num_sub_neigh = choose_sub_edges.size(0)
                sub_cat_att_feats = torch.cat((box_id_feats.expand([num_sub_neigh, -1]), choose_sub_rel_feats), 1)
                sub_atten = self.mp_atten_fc(sub_cat_att_feats)
                sub_alpha = F.softmax(sub_atten, dim=0)
                sub_feats = (sub_alpha * self.sub_rel_mp_fc(choose_sub_rel_feats)).sum(0)

                # object message from rel feats(may be null)
                choose_obj_edges_ind = torch.nonzero(im_i_edges[:, 2].contiguous() == box_id).squeeze()
                if choose_obj_edges_ind.size() == torch.Size([]):
                    box_id_feats = self.box_mp_fc(box_id_feats) + sub_feats
                    box_id_feats = F.relu(box_id_feats, inplace=True)
                    box_nodes_feats[box_id] = box_id_feats
                    continue
                choose_obj_edges = im_i_edges[choose_obj_edges_ind]
                choose_obj_rel_feats = im_i_rel_feats[choose_obj_edges_ind]
                box_id_feats = box_nodes_feats[box_id]

                # attention on object reltions
                num_obj_neigh = choose_obj_edges.size(0)
                obj_cat_att_feats = torch.cat((box_id_feats.expand([num_obj_neigh, -1]), choose_obj_rel_feats), 1)
                obj_atten = self.mp_atten_fc(obj_cat_att_feats)
                obj_alpha = F.softmax(obj_atten, dim=0)
                obj_feats = (obj_alpha * self.obj_rel_mp_fc(choose_obj_rel_feats)).sum(0)

                # add back to box feature
                box_id_feats = self.box_mp_fc(box_id_feats) + obj_feats + sub_feats
                box_id_feats = F.relu(box_id_feats, inplace=True)

                box_nodes_feats[box_id] = box_id_feats

        mp_box_feats = torch.stack(box_nodes_feats)
        return mp_box_feats

    @staticmethod
    def pad_sequence(
            inds,
            feats,
            rel_labels=None
    ):
        """pad lstm input and compute lstm length input
        e.g.
            batch0='abc', batch1='xy'
            output padded data:
                a b c
                x y *
            asterisk is padded data
            length is [2,2,1]
        Args:
            inds: torch.Tensor, image index of Rois, shape (NumOfRels, 3)
                e.g. inds[0,:]=[0,3,2] means relation of box 3 to box2 in image 0
            feats: torch.Tensor, unpadded features
                with shape of (NumOfRels, DimOfFeature)
            rel_labels: when train
        Returns:
            packed_lengths: list of int, length of pytorch RNN input
                refer to pytorch documentation(pack_padded_sequence)
            padded_feats: torch.Tensor, padded features
            re_inds: torch.Tensor, re-order image_inds, order by relation number descending order
            re_inds_np_inds: np.array, (NumOfRels,), the indexes in original rel_label refering to re_inds
        """
        num_img = int(inds[-1][0] + 1)
        feat_dim = feats.shape[-1]
        # length matrix:
        # Use structure array to sort length in descending order
        dtype = [('imgid', int), ('length', int), ('s', int), ('e', int)]
        values = list()
        im_start = 0
        for im_i in range(num_img):
            rel_ind_im_i, _ = np.where(inds[:, 0:1] == im_i)
            length_im_i = len(rel_ind_im_i)
            s = im_start
            e = im_start + length_im_i
            values.append((im_i, length_im_i, s, e))
            im_start = e
        length_matrix = np.array(values, dtype=dtype)
        length_matrix[::-1].sort(order='length')
        max_length = int(length_matrix[0][1])
        sorted_lengths = length_matrix['length']

        # get feat
        padding_feat = torch.FloatTensor(num_img, max_length, feat_dim).zero_()
        padding_inds = torch.LongTensor(num_img, max_length, inds.shape[-1]).zero_()
        if rel_labels is not None:
            padding_rel_labels = torch.LongTensor(num_img, max_length, rel_labels.shape[-1]).zero_()

        if feats.data.is_cuda:
            padding_feat = padding_feat.cuda(feats.get_device())
            padding_inds = padding_inds.cuda(feats.get_device())
            if rel_labels is not None:
                padding_rel_labels = padding_rel_labels.cuda(feats.get_device())
        # in re-ordered order
        for re_im_i in range(num_img):
            length_i = length_matrix[re_im_i][1]
            if length_i == 0:
                continue
            s = length_matrix['s'][re_im_i]
            e = length_matrix['e'][re_im_i]
            padding_feat[re_im_i][:length_i] = feats[s:e].data
            padding_inds[re_im_i][:length_i] = inds[s:e]
            if rel_labels is not None:
                padding_rel_labels[re_im_i][:length_i] = rel_labels[s:e].data

        re_inds_np_inds = np.array([], dtype='int')
        for ix, v in enumerate(length_matrix):
            s, e = v[2], v[3]
            re_inds_np_inds = np.concatenate(
                (re_inds_np_inds, np.arange(s, e)),
                axis=0
            )
        re_inds = inds[re_inds_np_inds.tolist()]

        if rel_labels is None:
            return Variable(padding_feat), sorted_lengths, re_inds, padding_inds
        else:
            return Variable(padding_feat), sorted_lengths, re_inds, padding_rel_labels

    @staticmethod
    def re_order_packed_seq(packed_seq, ori_inds, re_inds):
        """Re-order pack sequence
        original: [0,0,1,1,1,2,2,2,2]
        re_inds:[2,2,2,2,1,1,1,0,0]

        re order packed_seq by original order
        Args:
            packed_seq: PackedSequence
            ori_inds: torch.Tensor, original index, shape(NumOfSeq, 3)
            re_inds: torch.Tensor, current index, shape(NumOfSeq, 3)
        Returns:
            seq: torch.Tensor, re-order sequence
        """
        #embed(header='re_order')
        unpack, lens = pad_packed_sequence(packed_seq, batch_first=True)
        # unpack: (NumImg, MaxLength, DIM)
        # lens: (NumImg, 1)
        num_img_with_rels = len(lens)
        num_img = int(ori_inds[-1][0] + 1)
        # feats: remove padding from unpack
        # perm: permutation of feats
        if unpack.data.is_cuda:
            feats = torch.Tensor().cuda(unpack.get_device())
            perm = torch.LongTensor().cuda(unpack.get_device())
        else:
            feats = torch.Tensor()
            perm = torch.LongTensor()

        for im_i in range(num_img_with_rels):
            if feats.shape == torch.Size([]):
                feats = unpack[im_i][:lens[im_i]]
            else:
                feats = torch.cat([feats, unpack[im_i][:lens[im_i]]], dim=0)

        for im_i in range(num_img):
            re_im_i_inds = np.where(re_inds[:, 0:1] == im_i)[0]
            if re_im_i_inds.size == 0:
                continue
            if unpack.data.is_cuda:
                re_im_i_inds = torch.LongTensor(re_im_i_inds).cuda(unpack.get_device())
            else:
                re_im_i_inds = torch.LongTensor(re_im_i_inds)
            if perm.shape == torch.Size([]):
                perm = re_im_i_inds
            else:
                perm = torch.cat([perm, re_im_i_inds], dim=0)

        return feats[perm]

    def forward(
            self,
            x,
            im_sizes,
            image_offset,
            gt_boxes=None,
            gt_classes=None,
            gt_rels=None,
            proposals=None,
            train_anchor_inds=None,
            return_fmap=False
    ):
        """
        Forward pass for Relation detection
        Args:
            x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
            im_sizes: A numpy array of (h, w, scale) for each image.
            image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)

            parameters for training:
            gt_boxes: [num_gt, 4] GT boxes over the batch.
            gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
            gt_rels:
            proposals:
            train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
            return_fmap:

        Returns:
            If train:
                scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            If test:
                prob dists, boxes, img inds, maxscores, classes
        """
        s_t = time.time()
        verbose = False

        def check(sl, een, sst=s_t):
            if verbose:
                print('{}{}'.format(sl, een-sst))

        result = self.detector(
            x, im_sizes, image_offset, gt_boxes,
            gt_classes, gt_rels, proposals, train_anchor_inds, return_fmap=True
        )
        check('detector', tt())

        assert not result.is_none(), 'Empty detection result'

        # image_offset refer to Blob
        # self.batch_size_per_gpu * index
        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        obj_scores, box_classes = F.softmax(result.rm_obj_dists[:, 1:].contiguous(), dim=1).max(1)
        box_classes += 1
        # TODO: predcls implementation obj_scores and box_classes

        num_img = im_inds[-1] + 1

        # embed(header='rel_model.py before rel_assignments')
        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'

            # only in sgdet mode

            # shapes:
            # im_inds: (box_num,)
            # boxes: (box_num, 4)
            # rm_obj_labels: (box_num,)
            # gt_boxes: (box_num, 4)
            # gt_classes: (box_num, 2) maybe[im_ind, class_ind]
            # gt_rels: (rel_num, 4)
            # image_offset: integer
            result.rel_labels = rel_assignments(
                im_inds.data,
                boxes.data,
                result.rm_obj_labels.data,
                gt_boxes.data,
                gt_classes.data,
                gt_rels.data,
                image_offset,
                filter_non_overlap=True,
                num_sample_per_gt=1
            )
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)
        # union boxes feats (NumOfRels, obj_dim)
        union_box_feats = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:].contiguous())
        # single box feats (NumOfBoxes, feats)
        box_feats = self.obj_feature_map(result.fmap.detach(), rois)
        # box spatial feats (NumOfBox, 4)

        box_pair_feats = self.fuse_message(union_box_feats, boxes, box_classes, rel_inds)
        box_pair_score = self.relpn_fc(box_pair_feats)

        if self.training:
            # sampling pos and neg relations here for training
            rel_sample_pos, rel_sample_neg = 0, 0
            pn_rel_label, pn_pair_score = list(), list()
            for i, s, e in enumerate_by_image(result.rel_labels[:, 0].data.contiguous()):
                im_i_rel_label = result.rel_labels[s:e].contiguous()
                im_i_box_pair_score = box_pair_score[s:e].contiguous()

                im_i_rel_fg_inds = torch.nonzero(im_i_rel_label[:, -1].contiguous()).squeeze()
                im_i_rel_fg_inds = im_i_rel_fg_inds.data.cpu().numpy()
                im_i_fg_sample_num = min(RELEVANT_PER_IM, im_i_rel_fg_inds.shape[0])
                if im_i_rel_fg_inds.size > 0:
                    im_i_rel_fg_inds = np.random.choice(im_i_rel_fg_inds, size=im_i_fg_sample_num, replace=False)

                im_i_rel_bg_inds = torch.nonzero(im_i_rel_label[:, -1].contiguous() == 0).squeeze()
                im_i_rel_bg_inds = im_i_rel_bg_inds.data.cpu().numpy()
                im_i_bg_sample_num = min(EDGES_PER_IM - im_i_fg_sample_num, im_i_rel_bg_inds.shape[0])
                if im_i_rel_bg_inds.size > 0:
                    im_i_rel_bg_inds = np.random.choice(im_i_rel_bg_inds, size=im_i_bg_sample_num, replace=False)

                #print('{}/{} fg/bg in image {}'.format(im_i_fg_sample_num, im_i_bg_sample_num, i))
                rel_sample_pos += im_i_fg_sample_num
                rel_sample_neg += im_i_bg_sample_num

                im_i_keep_inds = np.append(im_i_rel_fg_inds, im_i_rel_bg_inds)
                im_i_pair_score = im_i_box_pair_score[im_i_keep_inds.tolist()].contiguous()

                im_i_rel_pn_labels = Variable(
                    torch.zeros(im_i_fg_sample_num + im_i_bg_sample_num).type(torch.LongTensor).cuda(x.get_device())
                )
                im_i_rel_pn_labels[:im_i_fg_sample_num] = 1

                pn_rel_label.append(im_i_rel_pn_labels)
                pn_pair_score.append(im_i_pair_score)

            result.rel_pn_dists = torch.cat(pn_pair_score, 0)
            result.rel_pn_labels = torch.cat(pn_rel_label, 0)
            result.rel_sample_pos = torch.Tensor([rel_sample_pos]).cuda(im_i_rel_label.get_device())
            result.rel_sample_neg = torch.Tensor([rel_sample_neg]).cuda(im_i_rel_label.get_device())

        box_pair_relevant = F.softmax(box_pair_score, dim=1)
        box_pos_pair_ind = torch.nonzero(
            box_pair_relevant[:, 1].contiguous() > box_pair_relevant[:, 0].contiguous()
        ).squeeze()

        if box_pos_pair_ind.data.shape == torch.Size([]):
            return None
        #print('{}/{} trim edges'.format(box_pos_pair_ind.size(0), rel_inds.size(0)))
        result.rel_trim_pos = torch.Tensor([box_pos_pair_ind.size(0)]).cuda(box_pos_pair_ind.get_device())
        result.rel_trim_total = torch.Tensor([rel_inds.size(0)]).cuda(rel_inds.get_device())

        if self.trim_graph:
            # filtering relations
            filter_rel_inds = rel_inds[box_pos_pair_ind.data]
            filter_box_pair_feats = box_pair_feats[box_pos_pair_ind.data]
        else:
            filter_rel_inds = rel_inds
            filter_box_pair_feats = box_pair_feats
        if self.training:
            if self.trim_graph:
                filter_rel_labels = result.rel_labels[box_pos_pair_ind.data]
            else:
                filter_rel_labels = result.rel_labels
            num_gt_filtered = torch.nonzero(filter_rel_labels[:, -1])
            if num_gt_filtered.shape == torch.Size([]):
                num_gt_filtered = 0
            else:
                num_gt_filtered = num_gt_filtered.size(0)
            num_gt_orignial = torch.nonzero(result.rel_labels[:, -1]).size(0)
            result.rel_pn_recall = torch.Tensor([num_gt_filtered / num_gt_orignial]).cuda(x.get_device())
            result.rel_labels = filter_rel_labels
        check('trim', tt())

        # message passing between boxes and relations
        if self.mode in ('sgcls', 'sgdet'):
            for _ in range(self.mp_iter_num):
                box_feats = self.message_passing(box_feats, filter_box_pair_feats, filter_rel_inds)
            box_cls_scores = self.cls_fc(box_feats)
            result.rm_obj_dists = box_cls_scores
            obj_scores, box_classes = F.softmax(box_cls_scores[:, 1:].contiguous(), dim=1).max(1)
            box_classes += 1  # skip background
        check('mp', tt())

        # RelationCNN
        filter_box_pair_feats_fc1 = self.relcnn_fc1(filter_box_pair_feats)
        filter_box_pair_score = self.relcnn_fc2(filter_box_pair_feats_fc1)

        result.rel_dists = filter_box_pair_score
        pred_scores_stage_one = F.softmax(result.rel_dists, dim=1).data

        # filter_box_pair_feats is to be added to memory
        if self.training:
            padded_filter_feats, pack_lengths, re_filter_rel_inds, padded_rel_labels = \
                self.pad_sequence(
                    filter_rel_inds,
                    filter_box_pair_feats_fc1,
                    rel_labels=result.rel_labels
                )
        else:
            padded_filter_feats, pack_lengths, re_filter_rel_inds, padded_rel_inds = \
                self.pad_sequence(
                    filter_rel_inds,
                    filter_box_pair_feats_fc1
                )

        # trimming zeros to avoid no rel in image
        trim_pack_lengths = np.trim_zeros(pack_lengths)
        trim_padded_filter_feats = padded_filter_feats[:trim_pack_lengths.shape[0]]
        packed_filter_feats = pack_padded_sequence(
            trim_padded_filter_feats, trim_pack_lengths, batch_first=True
        )
        if self.training:
            trim_padded_rel_labels = padded_rel_labels[:trim_pack_lengths.shape[0]]
            packed_rel_labels = pack_padded_sequence(
                trim_padded_rel_labels, trim_pack_lengths, batch_first=True
            )
            rel_mem_dists = self.mem_module(
                inputs=packed_filter_feats,
                rel_labels=packed_rel_labels
            )
            rel_mem_dists = self.re_order_packed_seq(rel_mem_dists, filter_rel_inds, re_filter_rel_inds)
            result.rel_mem_dists = rel_mem_dists
        else:
            trim_padded_rel_inds = padded_rel_inds[:trim_pack_lengths.shape[0]]
            packed_rel_inds = pack_padded_sequence(
                trim_padded_rel_inds, trim_pack_lengths, batch_first=True
            )
            rel_mem_dists = self.mem_module(
                inputs=packed_filter_feats,
                rel_inds=packed_rel_inds,
                obj_classes=box_classes
            )
            rel_mem_probs = self.re_order_packed_seq(rel_mem_dists, filter_rel_inds, re_filter_rel_inds)
            rel_mem_probs = rel_mem_probs.data

        check('mem', tt())
        if self.training:
            return result

        # pad stage one output in rel_mem_probs if it sums zero
        for rel_i in range(rel_mem_probs.size(0)):
            rel_i_probs = rel_mem_probs[rel_i]
            if rel_i_probs.sum() == 0:
                rel_mem_probs[rel_i] = pred_scores_stage_one[rel_i]

        """
        filter_dets
        boxes: bbox regression else [num_box, 4]
        obj_scores: [num_box] probabilities for the scores
        obj_classes: [num_box] class labels integer
        rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
        pred_scores: [num_rel, num_predicates] including irrelevant class(#relclass + 1)
        """
        check('mem processing', tt())
        return filter_dets(boxes, obj_scores, box_classes, filter_rel_inds[:, 1:].contiguous(), rel_mem_probs)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs


