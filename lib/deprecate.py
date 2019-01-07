
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