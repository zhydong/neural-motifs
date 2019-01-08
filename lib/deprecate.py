
# model v3: binary cross entropy sampling process
# choose the groud truth sample and randomly choose a negative sample


def fuck():
    bin_labels = [[0, 1] for _ in range(rel_mem_bin_dists.size(0))]
    rel_mem_bin_labels = torch.LongTensor(bin_labels)
    rel_mem_bin_labels = Variable(rel_mem_bin_labels.view(-1))
    if x.data.is_cuda:
        rel_mem_bin_labels = rel_mem_bin_labels.cuda(x.get_device())

    neg = []
    pos = []
    for rel_i in range(rel_mem_bin_dists.size(0)):
        rel_i_rel_labels = int(result.rel_labels[rel_i, 3])
        choose = [i for i in range(self.num_rels) if i != rel_i_rel_labels]
        choose_i = int(np.random.choice(choose))
        neg.append(choose_i)
        pos.append(rel_i_rel_labels)
    neg = torch.LongTensor(neg)
    neg = torch.cat((neg, neg), 0).view(-1, 1, 2)
    pos = torch.LongTensor(pos)
    pos = torch.cat((pos, pos), 0).view(-1, 1, 2)
    if x.data.is_cuda:
        neg = Variable(neg).cuda(x.get_device())
        pos = Variable(pos).cuda(x.get_device())

    rel_mem_bin_dists_sample = torch.cat(
        (
            rel_mem_bin_dists.gather(1, neg),
            rel_mem_bin_dists.gather(1, pos)
        ),
        1
    ).view(-1, 2)

def pad_sequencedeletedcode():
    # compute re_inds
    re_inds = torch.Tensor()
    if feats.data.is_cuda:
        re_inds = re_inds.cuda(feats.get_device())
    for ix, im_i in enumerate(length_matrix['imgid']):
        im_i_length = length_matrix[ix][1]
        im_i_inds = torch.ones(int(im_i_length)) * im_i.astype('float')
        if feats.data.is_cuda:
            re_inds = torch.cat([re_inds, im_i_inds.cuda(feats.get_device())], dim=0)
        else:
            re_inds = torch.cat([re_inds, im_i_inds], dim=0)


    re_inds_np_inds = np.array([], dtype='int')
    for ix, v in enumerate(length_matrix):
        s, e = v[2], v[3]
        re_inds_np_inds = np.concatenate(
            (re_inds_np_inds, np.arange(s, e)),
            axis=0
        )
    re_inds = inds[re_inds_np_inds.tolist()]
    embed(header='pad sequence')


def mem_processing():
    pred_scores_rel = []
    for rel_i in range(self.num_rels):
        rel_i_rel_mem_dists = rel_mem_dists[rel_i]
        rel_i_rel_mem_dists = self.re_order_packed_seq(
            rel_i_rel_mem_dists,
            filter_rel_inds,
            re_filter_rel_inds
        )
        rel_i_pred_scores = F.softmax(rel_i_rel_mem_dists, dim=1)
        pred_scores_rel.append(rel_i_pred_scores)
    pred_probs_rel = torch.stack(pred_scores_rel)
    # TODO:
    # multiply P(M|O)
    box0_class = box_classes[filter_rel_inds[:, 1]]
    box1_class = box_classes[filter_rel_inds[:, 2]]

    pred_probs_final = []
    for rel_i, (c0, c1) in enumerate(zip(box0_class, box1_class)):
        class_inds = c0.data * self.num_rels + c1.data
        c0c1_rel_distribution = self.rel_obj_distribution.weight.data[class_inds]
        if torch.nonzero(c0c1_rel_distribution).size() == torch.Size([]):
            # use first stage softmax
            pred_probs_final.append(pred_scores_stage_one[rel_i].data[None, :])
        else:
            rel_i_probs = pred_probs_rel[:, rel_i, :]
            # use memory output
            rel_i_pred_probs = torch.mm(c0c1_rel_distribution, rel_i_probs.data)
            pred_probs_final.append(rel_i_pred_probs)
    pred_probs_final = Variable(torch.cat(pred_probs_final, 0))
