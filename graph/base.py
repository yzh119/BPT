from graphbuilder import SegmentTree, FullyConnected, OpenAISparse
import torch as th

class GraphBatcher:
    def __init__(self, triu=False, graph_type='stt', **kwargs):
        self._graph_cache = {}
        self.triu = triu
        self.graph_type = graph_type
        self.attrs = kwargs

    def _get_graph(self, l):
        if l in self._graph_cache:
            return self._graph_cache[l]
        else:
            attrs = self.attrs
            if self.graph_type == 'openai':
                new_g = OpenAISparse(l, attrs['stride'], attrs['l'], triu=self.triu)
            elif self.graph_type == 'dense':
                new_g = FullyConnected(l, triu=self.triu, window=attrs.get('window', 8192))
            elif self.graph_type == 'stt':
                new_g = SegmentTree(l, triu=self.triu, step=attrs['step'], clip_dist=attrs['clip_dist'])
            else:
                raise KeyError('unrecognized graph type')
            self._graph_cache[l] = new_g
            return new_g

    def __call__(self, batch):
        raise NotImplementedError

class EncDecBatcher(GraphBatcher):
    def __init__(self, graph_type='stt', **kwargs):
        self._graph_cache_enc = {}
        self._graph_cache_dec = {}
        self.graph_type = graph_type
        self.attrs = kwargs

    def _get_graph(self, l):
        raise NotImplementedError

    def _get_graph_enc(self, l):
        if l in self._graph_cache_enc:
            return self._graph_cache_enc[l]
        else:
            attrs = self.attrs
            if self.graph_type == 'dense':
                new_g = FullyConnected(l, triu=False, window=attrs.get('window', 8192))
            elif self.graph_type == 'stt':
                new_g = SegmentTree(l, triu=False, step=attrs['step'], clip_dist=attrs['clip_dist'])
            else:
                raise KeyError('unrecognized graph type')
            self._graph_cache_enc[l] = new_g
            return new_g

    def _get_graph_dec(self, l):
        if l in self._graph_cache_dec:
            return self._graph_cache_dec[l]
        else:
            args = self.attrs
            if self.graph_type == 'dense':
                new_g = FullyConnected(l, triu=True, window=args.get('window', 8192))
            elif self.graph_type == 'stt':
                new_g = SegmentTree(l, triu=True, step=args['step'], clip_dist=args['clip_dist'])
            else:
                raise KeyError('unrecognized graph type')
            self._graph_cache_dec[l] = new_g
            return new_g

from graphbuilder import partition_csr

def get_csrs(g):
    out_csr = g.adjacency_matrix_scipy(transpose=True, fmt='csr', return_edge_ids=True)
    ROW, INDPTR_R = partition_csr(out_csr.indptr.astype('int64'))
    ROW = th.from_numpy(ROW)
    INDPTR_R = th.from_numpy(INDPTR_R)
    out_csr = (th.tensor(out_csr.indptr, dtype=th.long),
               th.tensor(out_csr.indices, dtype=th.long),
               th.tensor(out_csr.data, dtype=th.long))
    in_csr = g.adjacency_matrix_scipy(transpose=False, fmt='csr', return_edge_ids=True)
    COL, INDPTR_C = partition_csr(in_csr.indptr.astype('int64'))
    COL = th.from_numpy(COL)
    INDPTR_C = th.from_numpy(INDPTR_C)
    in_csr = (th.tensor(in_csr.indptr, dtype=th.long),
              th.tensor(in_csr.indices, dtype=th.long),
              th.tensor(in_csr.data, dtype=th.long))
    return out_csr, in_csr, ROW, INDPTR_R, COL, INDPTR_C

class Batch:
    def __init__(self, g=None, g_enc=None, g_dec=None, g_inter=None,
                 readout_ids=None, leaf_ids=None, y=None, nid_seg_arr=None,
                 n_nodes_per_dec_graph=None, n_sent_ctx=None):
        self.g = g
        self.g_enc = g_enc
        self.g_dec = g_dec
        self.g_inter = g_inter
        self.readout_ids = readout_ids
        self.leaf_ids = leaf_ids
        self.y = y
        self.nid_seg_arr = nid_seg_arr
        self.n_nodes_per_dec_graph = n_nodes_per_dec_graph
        self.n_sent_ctx = n_sent_ctx
        if g is not None:
            self.g_csr = get_csrs(g)
        if g_enc is not None:
            self.g_enc_csr = get_csrs(g_enc)
        if g_dec is not None:
            self.g_dec_csr = get_csrs(g_dec)
        if g_inter is not None:
            self.g_inter_csr = get_csrs(g_inter)
