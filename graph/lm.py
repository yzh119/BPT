from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from torchtext import datasets
from torchtext.datasets import LanguageModelingDataset
from .base import GraphBatcher, Batch
import numpy as np
import torch as th
import dgl


class Enwik8(LanguageModelingDataset):
    name = 'enwik8'
    dirname = 'enwik8'
    @classmethod
    def splits(cls, text_field, root='.', train='train.txt', validation='valid.txt', test='test.txt', **kwargs):
        return super(Enwik8, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

class Text8(LanguageModelingDataset):
    name = 'text8'
    dirname = 'text8'
    @classmethod
    def splits(cls, text_field, root='.', train='train.txt', validation='valid.txt', test='test.txt', **kwargs):
        return super(Text8, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

def get_lm_dataset(name='ptb'):
    if name == 'ptb':
        return datasets.PennTreebank
    elif name == 'wiki-2':
        return datasets.WikiText2
    elif name == 'wiki-103':
        return datasets.WikiText103
    elif name == 'enwik8':
        return Enwik8
    elif name == 'text8':
        return Text8
    else:
        raise KeyError('invalid dataset name')

class LMBatcher(GraphBatcher):
    def __init__(self, TEXT, graph_type='bpt', **kwargs):
        super(LMBatcher, self).__init__(triu=True, graph_type=graph_type, **kwargs)
        self.TEXT = TEXT
        self._cache = {}

    def __call__(self, batch):
        data = []
        labels = []

        v_shift, e_shift = 0, 0
        row, col = [], []
        leaf_ids, readout_ids = [], []
        pos_arr = []
        etypes = []

        for sent in batch:
            start = 0
            if isinstance(sent, tuple):
                sent_prev, sent = sent
                sent_prev = self.TEXT.numericalize([sent_prev]).view(-1)
                sent = self.TEXT.numericalize([sent]).view(-1)
                data.append(th.cat([sent_prev, sent[:-1]]))
                labels.append(sent[1:])
                length = len(sent_prev) + len(sent) - 1 
                start = len(sent_prev)
            else: 
                sent = self.TEXT.numericalize([sent]).view(-1)
                data.append(sent[:-1])
                labels.append(sent[1:])
                length = len(sent) - 1

            # get graph
            g = self._get_graph(length)
            # get pos
            pos_arr.append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            leaf_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift)))
            readout_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift, start=start)))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift)
            row.append(src)
            col.append(dst)
            etypes.append(th.from_numpy(etype))
            # update shift
            v_shift += g.number_of_nodes
            e_shift += g.number_of_edges

        n = v_shift
        leaf_ids = th.cat(leaf_ids)
        readout_ids = th.cat(readout_ids)
        pos_arr = th.cat(pos_arr)
        etypes = th.cat(etypes)
        row, col = map(np.concatenate, (row, col))
        coo = coo_matrix((np.zeros_like(row), (row, col)), shape=(n, n))
        g = dgl.DGLGraph(coo, readonly=True)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        data = th.cat(data)
        labels = th.cat(labels)
        g.edata['etype'] = etypes
        if self.graph_type == 'openai':
            g.ndata['pos_0'] = pos_arr // self.attrs['stride']
            g.ndata['pos_1'] = pos_arr % self.attrs['stride']
        else:
            g.ndata['pos'] = pos_arr

        g.nodes[leaf_ids].data['x'] = data

        return Batch(g=g, readout_ids=readout_ids, leaf_ids=leaf_ids, y=labels)


class LMDataset(Dataset):
    def __init__(self, lm_dataset, max_length=35, part=(0,1), test=False):
        n = len(lm_dataset[0].text)
        part_size = (n + part[1] - 1) // part[1]
        self.data = lm_dataset[0].text[part_size * part[0]: part_size * (part[0] + 1)]
        self.max_length = max_length
        self.test = test

    def __len__(self):
        return (len(self.data) + self.max_length // 2 - 1) // (self.max_length // 2)

    def __getitem__(self, index):
        if index == 0:
            return self.data[:self.max_length // 2]
        if self.test:
            return (self.data[(index - 1) * self.max_length // 2: index * self.max_length // 2],
                    self.data[index * self.max_length // 2: (index + 1) * self.max_length // 2])
        else: # training
            return self.data[(index - 1) * self.max_length // 2: (index + 1) * self.max_length // 2]



