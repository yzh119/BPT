from collections import namedtuple
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from torchtext import datasets
from .base import GraphBatcher, Batch
import numpy as np
import torch as th
import dgl


def get_text_classification_dataset(name='sst'):
    if name == 'sst1' or name == 'sst2':
        return datasets.SST
    elif name == 'imdb':
        return datasets.IMDB
    else:
        raise KeyError('invalid dataset name')

class TCBatcher(GraphBatcher):
    def __init__(self, TEXT, LABEL, graph_type='stt', **kwargs):
        super(TCBatcher, self).__init__(triu=True, graph_type=graph_type, **kwargs)
        self.TEXT = TEXT
        self.LABEL = LABEL
        self._cache = {}

    def __call__(self, batch):
        data = []
        labels = []

        v_shift, e_shift = 0, 0
        row, col = [], []
        root_ids, leaf_ids = [], []
        pos_arr = []
        etypes = []

        for text, label in batch:
            text = self.TEXT.numericalize([text]).view(-1)
            label = self.LABEL.numericalize([label]).view(-1)

            data.append(text)
            labels.append(label)
            length = len(text)
            # get graph
            g = self._get_graph(length)
            # get pos
            pos_arr.append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            root_ids.append(g.root_id(v_shift=v_shift))
            # gather roots
            leaf_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift)))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift)
            row.append(src)
            col.append(dst)
            etypes.append(th.from_numpy(etype))
            # update shift
            v_shift += g.number_of_nodes
            e_shift += g.number_of_edges

        n = v_shift
        root_ids = th.tensor(root_ids)
        leaf_ids = th.cat(leaf_ids)
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
        g.ndata['pos'] = pos_arr
        g.nodes[leaf_ids].data['x'] = data

        return Batch(g=g, readout_ids=root_ids, leaf_ids=leaf_ids, y=labels)


class TCDataset(Dataset):
    def __init__(self, tc_dataset, mode='sst1'):
        self.mode = mode
        if mode == 'sst2':
            Example = namedtuple('Example', ['text', 'label'])
            self.data = []
            for example in tc_dataset:
                if example.label != 'neutral':
                   self.data.append(Example(text=example.text, label=example.label))
        else:
            self.data = tc_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].text, self.data[index].label
