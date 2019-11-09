import torch.nn as nn
import numpy as np
import dgl.function as fn
import torch as th
from dgl.nn.pytorch.softmax import edge_softmax
from .op import *

class PositionwiseFeedForward(nn.Module):
    '''
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    '''
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(th.relu(self.w_1(x))))


class BPTBlock(nn.Module):
    MAX_ETYPE = 1000
    def __init__(self, dim_model, h, dim_ff, rel_pos=False, drop_h=0.1, drop_a=0.1):
        super(BPTBlock, self).__init__()
        self.dim_model = dim_model
        self.h = h
        self.dim_ff = dim_ff
        self.d_k = self.dim_model // self.h
        self.rel_pos = rel_pos
        self.drop_h = nn.Dropout(drop_h)
        self.drop_att = nn.Dropout(drop_a)
        self.norm_in = nn.LayerNorm(self.dim_model)
        self.norm_inter = nn.LayerNorm(self.dim_model)

        self.proj_q = nn.Linear(dim_model, self.d_k * self.h, bias=False)
        self.proj_k = nn.Linear(dim_model, self.d_k * self.h, bias=False)
        self.proj_v = nn.Linear(dim_model, self.d_k * self.h, bias=False)
        self.proj_o = nn.Linear(self.d_k * self.h, dim_model, bias=False)

        if self.rel_pos:
            self.embed_ak = nn.Embedding(self.MAX_ETYPE, self.d_k)

        self.ffn = nn.Sequential(
            PositionwiseFeedForward(self.dim_model, self.dim_ff, dropout=drop_h),
            nn.Dropout(drop_h)
        )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, g, csr):
        h = g.ndata['h'] # get pos embedding
        if self.rel_pos:
            g.edata['ak'] = self.embed_ak(g.edata['etype'])

        # get in and out csr
        out_csr, in_csr, ROW, INDPTR_R, COL, INDPTR_C = csr 
        # get queries, keys and values
        g.ndata['q'] = self.proj_q(h).view(-1, self.h, self.d_k)
        g.ndata['k'] = self.proj_k(h).view(-1, self.h, self.d_k)
        g.ndata['v'] = self.proj_v(h).view(-1, self.h, self.d_k)

        e = masked_mm(
            ROW, INDPTR_R, out_csr[2], out_csr[1], COL, INDPTR_C, in_csr[2], in_csr[1], g.ndata['k'], g.ndata['q'])

        e_rel = 0
        if self.rel_pos:
            e_rel = node_mul_edge(COL, INDPTR_C, in_csr[2], g.ndata['q'], g.edata['ak'])

        e = self.drop_att(sparse_softmax(COL, INDPTR_C, in_csr[2], (e_rel + e) / np.sqrt(self.d_k)))
        a = vec_spmm(
            COL, INDPTR_C, in_csr[2], in_csr[1],
            ROW, INDPTR_R, out_csr[2], out_csr[1],
            e, g.ndata['v']).view(-1, self.d_k * self.h)
        o = self.drop_h(self.proj_o(a))
        h = self.norm_in(h + o)
        h = self.norm_inter(h + self.ffn(h))
        g.ndata['h'] = h


class BPTMemBlock(nn.Module):
    MAX_ETYPE = 1000
    def __init__(self, dim_model, h, dim_ff, rel_pos=False, drop_h=0.1, drop_a=0.1):
        super(BPTMemBlock, self).__init__()
        self.dim_model = dim_model
        self.h = h
        self.dim_ff = dim_ff
        self.d_k = self.dim_model // self.h
        self.rel_pos = rel_pos
        self.drop_h = nn.ModuleList([nn.Dropout(drop_h) for _ in range(2)])
        self.drop_att = nn.ModuleList([nn.Dropout(drop_a) for _ in range(2)])
        self.norm_in = nn.ModuleList([nn.LayerNorm(self.dim_model) for _ in range(2)])
        self.norm_inter = nn.LayerNorm(self.dim_model)

        self.proj_q = nn.ModuleList([nn.Linear(dim_model, self.d_k * self.h, bias=False) for _ in range(2)])
        self.proj_k = nn.ModuleList([nn.Linear(dim_model, self.d_k * self.h, bias=False) for _ in range(2)])
        self.proj_v = nn.ModuleList([nn.Linear(dim_model, self.d_k * self.h, bias=False) for _ in range(2)])
        self.proj_o = nn.ModuleList([nn.Linear(self.d_k * self.h, dim_model, bias=False) for _ in range(2)])

        if self.rel_pos:
            self.embed_ak = nn.Embedding(self.MAX_ETYPE, self.d_k)

        self.ffn = nn.Sequential(
            PositionwiseFeedForward(self.dim_model, self.dim_ff, dropout=drop_h),
            nn.Dropout(drop_h)
        )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, g, csr, mem, csr_inter):
        # Part I: self-attention
        h = g.ndata['h']
        if self.rel_pos:
            g.edata['ak'] = self.embed_ak(g.edata['etype'])

        # get csr
        out_csr, in_csr, ROW, INDPTR_R, COL, INDPTR_C = csr
        # get queries, keys and values
        g.ndata['q'] = self.proj_q[0](h).view(-1, self.h, self.d_k)
        g.ndata['k'] = self.proj_k[0](h).view(-1, self.h, self.d_k)
        g.ndata['v'] = self.proj_v[0](h).view(-1, self.h, self.d_k)

        e = masked_mm(
            ROW, INDPTR_R, out_csr[2], out_csr[1], COL, INDPTR_C, in_csr[2], in_csr[1], g.ndata['k'], g.ndata['q'])

        e_rel = 0
        if self.rel_pos:
            e_rel = node_mul_edge(COL, INDPTR_C, in_csr[2], g.ndata['q'], g.edata['ak'])

        e = self.drop_att[0](sparse_softmax(COL, INDPTR_C, in_csr[2], (e_rel + e) / np.sqrt(self.d_k)))
        a = vec_spmm(
            COL, INDPTR_C, in_csr[2], in_csr[1],
            ROW, INDPTR_R, out_csr[2], out_csr[1],
            e, g.ndata['v']).view(-1, self.d_k * self.h)
        o = self.drop_h[0](self.proj_o[0](a))
        h = self.norm_in[0](h + o)

        # Part II: attention to memory
        # get csr
        out_csr, in_csr, ROW, INDPTR_R, COL, INDPTR_C = csr_inter
        
        if (h.shape[0] < mem.shape[0]):
            h = th.cat([h, h.new_zeros(mem.shape[0] - h.shape[0], *h.shape[1:])], 0)

        # get queries, keys
        q = self.proj_q[1](h).view(-1, self.h, self.d_k)
        k = self.proj_k[1](mem).view(-1, self.h, self.d_k)
        v = self.proj_v[1](mem).view(-1, self.h, self.d_k)

        e = masked_mm(
            ROW, INDPTR_R, out_csr[2], out_csr[1], COL, INDPTR_C, in_csr[2], in_csr[1], k, q)
        e = self.drop_att[1](sparse_softmax(COL, INDPTR_C, in_csr[2], e / np.sqrt(self.d_k)))
        a = vec_spmm(
            COL, INDPTR_C, in_csr[2], in_csr[1],
            ROW, INDPTR_R, out_csr[2], out_csr[1],
            e, v).view(-1, self.d_k * self.h)
        # this would be deprecated after using bipartite graph.
        h = h[:g.number_of_nodes()]
        a = a[:g.number_of_nodes()]

        o = self.drop_h[1](self.proj_o[1](a))
        h = self.norm_in[1](h + o)

        # FFN
        h = self.norm_inter(h + self.ffn(h))
        g.ndata['h'] = h

    def infer(self,
              g,
              nids_eq_pos,
              eids_eq_pos,
              nids_eq_pos_leaf,
              g_inter, readout_ids):
        # Part I: self-attention
        h = g.nodes[nids_eq_pos].data['h']
        if self.rel_pos:
            g.edges[eids_eq_pos].data['ak'] = self.embed_ak(g.edges[eids_eq_pos].data['etype'])

        g.nodes[nids_eq_pos].data['q'] = self.proj_q[0](h).view(-1, self.h, self.d_k)
        g.nodes[nids_eq_pos].data['k'] = self.proj_k[0](h).view(-1, self.h, self.d_k)
        g.nodes[nids_eq_pos].data['v'] = self.proj_v[0](h).view(-1, self.h, self.d_k)

        g.apply_edges(
            lambda edges: {'e': (edges.src['k'] * edges.dst['q']).sum(dim=-1, keepdim=True)},
            eids_eq_pos
        )
        e = g.edges[eids_eq_pos].data['e']
        # relative positional encoding
        if self.rel_pos:
            g.apply_edges(
                lambda edges: {'e_rel': (edges.data['ak'].unsqueeze(1) * edges.dst['q']).sum(dim=-1, keepdim=True)},
                eids_eq_pos
            )
            e = e + g.edges[eids_eq_pos].data['e_rel']
        # softmax
        g.edges[eids_eq_pos].data['a'] = self.drop_att[0](
            edge_softmax(
                g, e / np.sqrt(self.d_k),
                eids_eq_pos
            )
        )
        # spmm
        g.send_and_recv(eids_eq_pos, fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'o'))
        o = g.nodes[nids_eq_pos].data['o'].view(-1, self.d_k * self.h)
        o = self.drop_h[0](self.proj_o[0](o))
        g.nodes[nids_eq_pos].data['h'] = self.norm_in[0](h + o)

        # Part II: attend to memory
        h = g.nodes[nids_eq_pos_leaf].data['h']
        q = self.proj_q[1](h).view(-1, self.h, self.d_k)
        g_inter.nodes[readout_ids].data['q'] = q
        g_inter.apply_edges(
            lambda edges: {'e': (edges.src['k'] * edges.dst['q']).sum(dim=-1, keepdim=True)})
        # softmax
        g_inter.edata['a'] = self.drop_att[1](edge_softmax(g_inter, g_inter.edata['e'] / np.sqrt(self.d_k)))
        # spmm
        g_inter.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'o'))
        o = g_inter.nodes[readout_ids].data['o'].view(-1, self.d_k * self.h)
        o = self.drop_h[1](self.proj_o[1](o))
        g.nodes[nids_eq_pos_leaf].data['h'] = h + o
        h = self.norm_in[1](g.nodes[nids_eq_pos].data['h'])

        # FFN
        h = self.norm_inter(h + self.ffn(h))
        g.nodes[nids_eq_pos].data['h'] = h

