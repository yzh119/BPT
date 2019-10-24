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
