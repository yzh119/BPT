from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from torchtext.datasets import TranslationDataset
from torchtext.vocab import Vocab
from .base import Batch, EncDecBatcher
import numpy as np
import torch as th
import dgl
import os
import collections

def get_mt_dataset(name='wmt14'):
    if name == 'wmt14':
        return WMT14
    elif name == 'iwslt':
        return IWSLT2015
    elif name == 'multi30k':
        return Multi30k
    else:
        raise KeyError('invalid dataset name')

class IWSLT2015(TranslationDataset):
    name = 'iwslt'
    dirname = '.'
    @classmethod
    def splits(cls, exts, fields, root='.', train='corpus', validation='IWSLT15.TED.dev2010',
               test='IWSLT15.TED.tst', **kwargs):
        data_splits = super(IWSLT2015, cls).splits(
            exts=exts, fields=fields, root=root,
            train=train, validation=validation, test=test)
        doc_index_splits = []
        for segment in [train, validation, test]:
            with open(os.path.join(os.path.join(root, cls.name), segment + '.doc'), 'r') as f:
                doc_index_splits.append([int(idx.strip()) for idx in f.readlines()])
        return zip(data_splits, doc_index_splits)

    @classmethod
    def load_vocab(cls, vocab_name=['vocab.zh', 'vocab.en'], root='.'):
        if not isinstance(vocab_name, list) or isinstance(vocab_name, tuple):
            vocab_name = [vocab_name]
        rst_vocab = []
        for vocab_name_i in vocab_name:
            counter = collections.Counter()
            with open(os.path.join(os.path.join(root, cls.name), vocab_name_i), 'r') as f:
                for word in f.readlines():
                    word = word.strip()
                    counter.update([word])
            rst_vocab.append(Vocab(counter, specials=[DocumentMTDataset.BOS_TOKEN,
                                                      DocumentMTDataset.EOS_TOKEN,
                                                      Vocab.UNK]))
        return rst_vocab


class WMT14(TranslationDataset):
    name = 'wmt14'
    dirname = '.'
    @classmethod
    def splits(cls, exts, fields, root='.',
               train='train.tok.clean.bpe.32000',
               validation='newstest2013.tok.bpe.32000',
               test='newstest2014.tok.bpe.32000.ende',
               **kwargs):
        return super(WMT14, cls).splits(
            exts=exts, fields=fields, root=root,
            train=train, validation=validation, test=test)

    @classmethod
    def load_vocab(cls, vocab_name=['vocab.bpe.32000'], root='.'):
        if not isinstance(vocab_name, list) or isinstance(vocab_name, tuple):
            vocab_name = [vocab_name]
        rst_vocab = []
        for vocab_name_i in vocab_name:
            counter = collections.Counter()
            with open(os.path.join(os.path.join(root, cls.name), vocab_name_i), 'r') as f:
                for word in f.readlines():
                    word = word.strip()
                    counter.update([word])
            rst_vocab.append(Vocab(counter, specials=[MTDataset.BOS_TOKEN,
                                                      MTDataset.EOS_TOKEN,
                                                      Vocab.UNK]))
        return rst_vocab

class Multi30k(TranslationDataset):
    name = 'multi30k'
    dirname ='.'
    @classmethod
    def splits(cls, exts, fields, root='.',
               train='train',
               validation='val',
               test='test2016',
               **kwargs):
        return super(Multi30k, cls).splits(
            exts=exts, fields=fields, root=root,
            train=train, validation=validation, test=test)

    @classmethod
    def load_vocab(cls, vocab_name=['vocab.en', 'vocab.de'], root='.'):
        if not isinstance(vocab_name, list) or isinstance(vocab_name, tuple):
            vocab_name = [vocab_name]
        rst_vocab = []
        for vocab_name_i in vocab_name:
            counter = collections.Counter()
            with open(os.path.join(os.path.join(root, cls.name), vocab_name_i), 'r') as f:
                for word in f.readlines():
                    word = word.strip()
                    counter.update([word])
            rst_vocab.append(Vocab(counter, specials=[MTDataset.BOS_TOKEN,
                                                      MTDataset.EOS_TOKEN,
                                                      Vocab.UNK]))
        return rst_vocab

class MTInferBatcher(EncDecBatcher):
    def __init__(self, TEXT, max_length, bos_token, graph_type='stt', **kwargs):
        super(MTInferBatcher, self).__init__(graph_type=graph_type, **kwargs)
        self.TEXT = TEXT
        self._cache = {}
        self.max_length = max_length
        self.bos_token = bos_token
        self.k = kwargs.get('k', 5)

    def __call__(self, batch):
        data = {'enc': [], 'dec': []}

        v_shift, e_shift = {'enc': 0, 'dec': 0}, {'enc': 0, 'dec': 0}
        row, col = {'enc': [], 'dec': []}, {'enc': [], 'dec': []}
        leaf_ids, dec_bos_ids = {'enc': [], 'dec': []}, []
        pos_arr = {'enc': [], 'dec': []}
        etypes = {'enc': [], 'dec': []}
        nid_seg_arr = []
        n_nodes_per_dec_graph = 0
        n_sent_ctx = []

        for src, _, n_sent_ctx_i in batch:
            if isinstance(src, tuple):
                src_text, src_seg = src
            else:
                src_text = src
                src_seg = [0, len(src_text)]

            trg_text = [self.bos_token]

            if isinstance(self.TEXT, tuple) or isinstance(self.TEXT, list):
                src_id = self.TEXT[0].numericalize([src_text]).view(-1)
                trg_id = self.TEXT[1].numericalize([trg_text]).view(-1)
            else:
                src_id = self.TEXT.numericalize([src_text]).view(-1)
                trg_id = self.TEXT.numericalize([trg_text]).view(-1)

            data['enc'].append(src_id)
            data['dec'].append(trg_id) # only for bos token

            # building src graph
            length = len(src_id)
            # get graph
            g = self._get_graph_enc(length)
            # get pos
            pos_arr['enc'].append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            leaf_ids['enc'].append(th.from_numpy(g.leaf_ids(v_shift=v_shift['enc'])))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift['enc'])
            row['enc'].append(src)
            col['enc'].append(dst)
            etypes['enc'].append(th.from_numpy(etype))
            # update_shift
            nid_seg = [
                np.arange(src_seg[i - 1] + v_shift['enc'], src_seg[i] + v_shift['enc'])
                for i in range(1, len(src_seg))
            ]
            nid_seg_arr.append(nid_seg)
            v_shift['enc'] += g.number_of_nodes
            e_shift['enc'] += g.number_of_edges

            # building trg graph
            length = self.max_length
            # get graph
            g = self._get_graph_dec(length)
            # get pos
            pos_arr['dec'].append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            dec_bos_ids.append(v_shift['dec'])
            leaf_ids['dec'].append(th.from_numpy(g.leaf_ids(v_shift=v_shift['dec'])))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift['dec'])
            row['dec'].append(src)
            col['dec'].append(dst)
            etypes['dec'].append(th.from_numpy(etype))
            # update_shift
            v_shift['dec'] += g.number_of_nodes
            e_shift['dec'] += g.number_of_edges
            n_nodes_per_dec_graph = g.number_of_nodes
            # update number of sentences in context. 
            n_sent_ctx.append(n_sent_ctx_i)

        # construct encoder
        n_enc = v_shift['enc']
        leaf_ids['enc'] = th.cat(leaf_ids['enc'])
        pos_arr['enc'] = th.cat(pos_arr['enc'])
        etypes['enc'] = th.cat(etypes['enc'])
        row['enc'], col['enc'] = map(np.concatenate, (row['enc'], col['enc']))
        coo_enc = coo_matrix((np.zeros_like(row['enc']), (row['enc'], col['enc'])), shape=(n_enc, n_enc))
        g_enc = dgl.DGLGraph(coo_enc, readonly=True)

        # construct decoder
        n_dec = v_shift['dec']
        leaf_ids['dec'] = th.cat(leaf_ids['dec'])
        pos_arr['dec'] = th.cat(pos_arr['dec'])
        etypes['dec'] = th.cat(etypes['dec'])
        row['dec'], col['dec'] = map(np.concatenate, (row['dec'], col['dec']))
        coo_dec = coo_matrix((np.zeros_like(row['dec']), (row['dec'], col['dec'])), shape=(n_dec, n_dec))
        g_dec = [dgl.DGLGraph(coo_dec, readonly=True) for _ in range(self.k)]

        # intialize graph
        g_enc.set_n_initializer(dgl.init.zero_initializer)
        g_enc.set_e_initializer(dgl.init.zero_initializer)
        for i in range(self.k):
            g_dec[i].set_n_initializer(dgl.init.zero_initializer)
            g_dec[i].set_e_initializer(dgl.init.zero_initializer)

        data['enc'] = th.cat(data['enc'])
        data['dec'] = th.cat(data['dec'])
        dec_bos_ids = th.LongTensor(dec_bos_ids)

        # assign enc graph feature
        g_enc.edata['etype'] = etypes['enc']
        g_enc.ndata['pos'] = pos_arr['enc']
        g_enc.nodes[leaf_ids['enc']].data['x'] = data['enc']
        # assign dec graph feature
        for i in range(self.k):
            g_dec[i].edata['etype'] = etypes['dec']
            g_dec[i].ndata['pos'] = pos_arr['dec']
            g_dec[i].nodes[dec_bos_ids].data['x'] = data['dec']

        return Batch(g_enc=g_enc, g_dec=g_dec, leaf_ids=leaf_ids, nid_seg_arr=nid_seg_arr,
                     n_nodes_per_dec_graph=n_nodes_per_dec_graph, n_sent_ctx=n_sent_ctx)

class MTBatcher(EncDecBatcher):
    def __init__(self, TEXT, graph_type='stt', **kwargs):
        super(MTBatcher, self).__init__(graph_type=graph_type, **kwargs)
        self.TEXT = TEXT
        self._cache = {}

    def __call__(self, batch):
        data = {'enc': [], 'dec': []}
        labels = []

        v_shift, e_shift = {'enc': 0, 'dec': 0}, {'enc': 0, 'dec': 0}
        row, col = {'enc': [], 'dec': []}, {'enc': [], 'dec': []}
        leaf_ids, readout_ids = {'enc': [], 'dec': []}, []
        pos_arr = {'enc': [], 'dec': []}
        etypes = {'enc': [], 'dec': []}
        row_inter, col_inter = [], []

        for src, trg, _ in batch:
            if isinstance(src, tuple):
                src_text, src_seg = src
                trg_text, trg_seg = trg
            else:
                src_text = src
                src_seg = [0, len(src_text)]
                trg_text = trg
                trg_seg = [0, len(trg_text) - 1]

            if isinstance(self.TEXT, tuple) or isinstance(self.TEXT, list):
                src_id = self.TEXT[0].numericalize([src_text]).view(-1)
                trg_id = self.TEXT[1].numericalize([trg_text]).view(-1)
            else:
                src_id = self.TEXT.numericalize([src_text]).view(-1)
                trg_id = self.TEXT.numericalize([trg_text]).view(-1)

            data['enc'].append(src_id)
            data['dec'].append(trg_id[:-1])
            labels.append(trg_id[1:])

            # building src graph
            length = len(src_id)
            # get graph
            g = self._get_graph_enc(length)
            # get pos
            pos_arr['enc'].append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            leaf_ids['enc'].append(th.from_numpy(g.leaf_ids(v_shift=v_shift['enc'])))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift['enc'])
            row['enc'].append(src)
            col['enc'].append(dst)
            etypes['enc'].append(th.from_numpy(etype))
            # update shift
            nid_src = [
                np.arange(v_shift['enc'] + src_seg[i - 1], v_shift['enc'] + src_seg[i])
                for i in range(1, len(src_seg))
            ]
            v_shift['enc'] += g.number_of_nodes
            e_shift['enc'] += g.number_of_edges

            # buildling trg graph
            length = len(trg_id) - 1
            # get graph
            g = self._get_graph_dec(length)
            # get pos
            pos_arr['dec'].append(th.from_numpy(g.get_pos()))
            # gather leaf nodes
            leaf_ids['dec'].append(th.from_numpy(g.leaf_ids(v_shift=v_shift['dec'])))
            readout_ids.append(th.arange(v_shift['dec'], v_shift['dec'] + length))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift['dec'])
            row['dec'].append(src)
            col['dec'].append(dst)
            etypes['dec'].append(th.from_numpy(etype))
            # update shift
            nid_trg = [
                np.arange(v_shift['dec'] + trg_seg[i - 1], v_shift['dec'] + trg_seg[i])
                for i in range(1, len(trg_seg))
            ]
            v_shift['dec'] += g.number_of_nodes
            e_shift['dec'] += g.number_of_edges

            # building inter graph
            for nid_src_i, nid_trg_i in zip(nid_src, nid_trg):
                row_inter.append(np.repeat(nid_src_i, len(nid_trg_i)))
                col_inter.append(np.tile(nid_trg_i, len(nid_src_i)))

        # construct encoder
        n_enc = v_shift['enc']
        leaf_ids['enc'] = th.cat(leaf_ids['enc'])
        pos_arr['enc'] = th.cat(pos_arr['enc'])
        etypes['enc'] = th.cat(etypes['enc'])
        row['enc'], col['enc'] = map(np.concatenate, (row['enc'], col['enc']))
        coo_enc = coo_matrix((np.zeros_like(row['enc']), (row['enc'], col['enc'])), shape=(n_enc, n_enc))
        g_enc = dgl.DGLGraph(coo_enc, readonly=True)

        # construct decoder
        n_dec = v_shift['dec']
        leaf_ids['dec'] = th.cat(leaf_ids['dec'])
        pos_arr['dec'] = th.cat(pos_arr['dec'])
        etypes['dec'] = th.cat(etypes['dec'])
        row['dec'], col['dec'] = map(np.concatenate, (row['dec'], col['dec']))
        coo_dec = coo_matrix((np.zeros_like(row['dec']), (row['dec'], col['dec'])), shape=(n_dec, n_dec))
        g_dec = dgl.DGLGraph(coo_dec, readonly=True)

        # construct inter-graph
        # the code here is ugly and we should replace it with DGL bipartite graph
        # in the future.
        n_inter = max(n_enc, n_dec)
        row_inter, col_inter = map(np.concatenate, (row_inter, col_inter))
        coo_inter = coo_matrix((np.zeros_like(row_inter), (row_inter, col_inter)), shape=(n_inter, n_inter))
        g_inter = dgl.DGLGraph(coo_inter, readonly=True)

        # process readout ids
        readout_ids = th.cat(readout_ids)

        # initialize graph
        g_enc.set_n_initializer(dgl.init.zero_initializer)
        g_enc.set_e_initializer(dgl.init.zero_initializer)
        g_dec.set_n_initializer(dgl.init.zero_initializer)
        g_dec.set_e_initializer(dgl.init.zero_initializer)
        g_inter.set_n_initializer(dgl.init.zero_initializer)
        g_inter.set_n_initializer(dgl.init.zero_initializer)

        data['enc'] = th.cat(data['enc'])
        data['dec'] = th.cat(data['dec'])
        labels = th.cat(labels)

        # assign enc graph feature
        g_enc.edata['etype'] = etypes['enc']
        g_enc.ndata['pos'] = pos_arr['enc']
        g_enc.nodes[leaf_ids['enc']].data['x'] = data['enc']
        # assign dec graph feature
        g_dec.edata['etype'] = etypes['dec']
        g_dec.ndata['pos'] = pos_arr['dec']
        g_dec.nodes[leaf_ids['dec']].data['x'] = data['dec']

        return Batch(g_enc=g_enc, g_dec=g_dec, g_inter=g_inter, readout_ids=readout_ids, leaf_ids=leaf_ids, y=labels)


class MTDataset(Dataset):
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    def __init__(self, mt_dataset, part=(0,1)):
        examples = mt_dataset.examples
        n = len(examples)
        part_size = (n + part[1] - 1) // part[1]
        self.data = examples[part_size * part[0]: part_size * (part[0] + 1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].src + [self.EOS_TOKEN], [self.BOS_TOKEN] + self.data[index].trg + [self.EOS_TOKEN], 0


class DocumentMTDataset(MTDataset):
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    def __init__(self, mt_dataset, context_length=1024, part=(0,1)):
        examples = mt_dataset[0].examples
        n = len(examples)
        part_size = (n + part[1] - 1) // part[1]
        self.examples = examples[part_size * part[0]: part_size * (part[0] + 1)]
        doc_indices = mt_dataset[1] + [n]
        self.doc_index = doc_indices[part_size * part[0]: part_size * (part[0] + 1) + 1]

        self.doc_src = []
        self.doc_trg = []
        self.seg_src = []
        self.seg_trg = []
        self.n_sent_ctx = []
        for i in range(len(self.doc_index) - 1):
            length = 0
            segments = [[]]
            for j in range(self.doc_index[i], self.doc_index[i + 1]):
                length += len(self.examples[j].src) + 1 # one for eos
                segments[-1].append(j)
                if length >= context_length // 2:
                    length = 0
                    segments.append([])
            if len(segments[-1]) == 0:
                segments.pop()
        
            for j in range(len(segments)):
                doc_src_i = []
                doc_trg_i = [self.BOS_TOKEN]
                seg_src_i = [0]
                seg_trg_i = [0]
                if j > 0 and context_length > 0:
                    for k in segments[j-1]:
                        doc_src_i.extend(self.examples[k].src + [self.EOS_TOKEN])
                        doc_trg_i.extend(self.examples[k].trg + [self.EOS_TOKEN])
                        seg_src_i.append(len(doc_src_i))
                        seg_trg_i.append(len(doc_trg_i) - 1)
                    self.n_sent_ctx.append(len(segments[j-1]))
                else:
                    self.n_sent_ctx.append(0)

                for k in segments[j]:
                    doc_src_i.extend(self.examples[k].src + [self.EOS_TOKEN])
                    doc_trg_i.extend(self.examples[k].trg + [self.EOS_TOKEN])
                    seg_src_i.append(len(doc_src_i))
                    seg_trg_i.append(len(doc_trg_i) - 1)

                self.doc_src.append(doc_src_i)
                self.doc_trg.append(doc_trg_i)
                self.seg_src.append(seg_src_i)
                self.seg_trg.append(seg_trg_i)

    def __len__(self):
        return len(self.doc_src)

    def __getitem__(self, index):
        return (self.doc_src[index], self.seg_src[index]), (self.doc_trg[index], self.seg_trg[index]), self.n_sent_ctx[index]
