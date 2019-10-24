from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from torchtext import datasets
from .base import GraphBatcher, Batch
import numpy as np
import torch as th
import dgl


def get_nli_dataset(name='snli'):
    if name == 'snli':
        return datasets.SNLI
    elif name == 'mnli':
        return datasets.MultiNLI
    else:
        raise KeyError('invalid dataset name')


class NLIBatcher(GraphBatcher):
    def __init__(self, TEXT, LABEL, graph_type='stt', **kwargs):
        super(NLIBatcher, self).__init__(triu=True, graph_type=graph_type, **kwargs)
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
        row_inter, col_inter = [], []

        for premise, hypo, label in batch:
            premise = self.TEXT.numericalize([premise]).view(-1)
            hypo = self.TEXT.numericalize([hypo]).view(-1)
            label = self.LABEL.numericalize([label]).view(-1)

            data.append(th.cat([premise, hypo], -1))
            labels.append(label)

            # building premise graph
            length = len(premise)
            # get graph
            g = self._get_graph(length)
            # get pos
            pos_arr.append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            root_ids.append(g.root_id(v_shift=v_shift))
            leaf_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift)))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift)
            row.append(src)
            col.append(dst)
            etypes.append(th.from_numpy(etype))
            # update shift
            nid_premise_leaf = np.arange(v_shift, v_shift + length)
            v_shift += g.number_of_nodes
            e_shift += g.number_of_edges

            # building hypo graph
            length = len(hypo)
            # get graph
            g = self._get_graph(length)
            # get pos
            pos_arr.append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            root_ids.append(g.root_id(v_shift=v_shift))
            leaf_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift)))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift)
            row.append(src)
            col.append(dst)
            etypes.append(th.from_numpy(etype))
            # update shift
            nid_hypo_leaf = np.arange(v_shift, v_shift + length)
            v_shift += g.number_of_nodes
            e_shift += g.number_of_edges

            # building inter graph
            row_inter.append(np.repeat(nid_premise_leaf, len(nid_hypo_leaf)))
            col_inter.append(np.tile(nid_hypo_leaf, len(nid_premise_leaf)))
            row_inter.append(np.repeat(nid_hypo_leaf, len(nid_premise_leaf)))
            col_inter.append(np.tile(nid_premise_leaf, len(nid_hypo_leaf)))

        n = v_shift
        root_ids = th.tensor(root_ids)
        leaf_ids = th.cat(leaf_ids)
        pos_arr = th.cat(pos_arr)
        etypes = th.cat(etypes)
        row, col = map(np.concatenate, (row, col))
        row_inter, col_inter = map(np.concatenate, (row_inter, col_inter))
        coo = coo_matrix((np.zeros_like(row), (row, col)), shape=(n, n))
        g = dgl.DGLGraph(coo, readonly=True)
        coo_inter = coo_matrix((np.zeros_like(row_inter), (row_inter, col_inter)), shape=(n, n))
        g_inter = dgl.DGLGraph(coo_inter, readonly=True)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)
        g_inter.set_n_initializer(dgl.init.zero_initializer)
        g_inter.set_e_initializer(dgl.init.zero_initializer)

        data = th.cat(data)
        labels = th.cat(labels)
        g.edata['etype'] = etypes
        g.ndata['pos'] = pos_arr
        g.nodes[leaf_ids].data['x'] = data

        return Batch(g=g, g_inter=g_inter, readout_ids=root_ids, leaf_ids=leaf_ids, y=labels)


class NLIDataset(Dataset):
    def __init__(self, nli_dataset):
        self.data = nli_dataset.examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].premise, self.data[index].hypothesis, self.data[index].label
