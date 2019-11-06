"""
Beam search module for Document Level Machine Translation.
"""
from .model import SegmentTreeEncoderDecoder
import torch as th
import numpy as np
import dgl
import copy
import itertools


def move_to_device(_tuple, device):
    return tuple(x.to(device) if isinstance(x, th.Tensor) else move_to_device(x, device) for x in _tuple)

class SegmentTreeBeamSearch(SegmentTreeEncoderDecoder):
    BEAM_SIZE = 5
    def __init__(self, vocab_sizes, dim_model, dim_ff, h,
                 n_layers, m_layers,
                 dropouti=0.1, dropouth=0.1, dropouta=0.1, dropoutc=0, rel_pos=False):
        super(SegmentTreeBeamSearch, self).__init__(vocab_sizes, dim_model, dim_ff, h,
                                                    n_layers, m_layers,
                                                    dropouti=dropouti, dropouth=dropouth,
                                                    dropoutc=dropoutc, dropouta=dropouta, rel_pos=rel_pos)

    def forward(self, batch, eos_id, sent_max_len=64):
        """

        Parameters
        ----------
        batch :
        sent_max_len : int
            The maximum length per sentence.
        """
        batch_size = len(batch.nid_seg_arr)
        g_enc = batch.g_enc.local_var()
        device = next(self.parameters()).device
        csr_enc = move_to_device(batch.g_enc_csr, device)

        leaf_ids = batch.leaf_ids['enc']
        h_enc = self.embed[0](g_enc.nodes[leaf_ids].data['x'])
        if not self.rel_pos:
            h_enc += self.pos_enc(g_enc.nodes[leaf_ids].data['pos'])
        g_enc.nodes[leaf_ids].data['h'] = self.norm(self.emb_dropout(h_enc))

        for enc_layer in self.enc_layers:
            enc_layer(g_enc, csr_enc)

        mem = g_enc.ndata['h']
        mem_key, mem_value = [], []
        for dec_layer in self.dec_layers:
            mem_key.append(dec_layer.proj_k[1](mem).view(-1, dec_layer.h, dec_layer.d_k))
            mem_value.append(dec_layer.proj_v[1](mem).view(-1, dec_layer.h, dec_layer.d_k))

        g_dec = [batch.g_dec[k].local_var() for k in range(self.BEAM_SIZE)]
        leaf_ids = batch.leaf_ids['dec']
        outputs = [[] for _ in range(self.BEAM_SIZE)]
        sent_indices = [[0 for _ in range(self.BEAM_SIZE)] for _ in range(batch_size)]
        y = [[[] for _ in range(self.BEAM_SIZE)] for _ in range(batch_size)]
        score = [[0. for _ in range(self.BEAM_SIZE)] for _ in range(batch_size)]
        start_pos = [[0 for _ in range(self.BEAM_SIZE)] for _ in range(batch_size)]
        eos = [[False for _ in range(self.BEAM_SIZE)] for _ in range(batch_size)]
        stop = th.zeros(batch_size).long()

        for step in itertools.count():
            tmp_k, tmp_v = [[] for _ in range(self.BEAM_SIZE)], [[] for _ in range(self.BEAM_SIZE)]
            for j in range(self.BEAM_SIZE):
                nids_lt_pos = g_dec[j].filter_nodes(lambda v: (v.data['pos'] < step)).cpu()
                nids_eq_pos = g_dec[j].filter_nodes(lambda v: (v.data['pos'] == step)).cpu()
                eids_eq_pos = g_dec[j].filter_edges(lambda e: (e.dst['pos'] == step)).cpu()
                nids_eq_pos_leaf = g_dec[j].filter_nodes(lambda v: (v.data['pos'] == step), leaf_ids).cpu()
                nids_le_pos = g_dec[j].filter_nodes(lambda v: (v.data['pos'] <= step)).cpu()

                h_dec = self.embed[-1](g_dec[j].nodes[nids_eq_pos_leaf].data['x'])
                if not self.rel_pos:
                    h_dec += self.pos_enc(g_dec[j].nodes[nids_eq_pos_leaf].data['pos'])
                g_dec[j].nodes[nids_eq_pos_leaf].data['h'] = self.norm(self.emb_dropout(h_dec))

                g_inter = []
                readout_ids = []
                mem_ids = []
                n_inter = 0
                for i in range(batch_size):
                    g_inter_i = dgl.DGLGraph()
                    segment = batch.nid_seg_arr[i][sent_indices[i][j]]
                    n = len(segment)
                    g_inter_i.add_nodes(n + 1)
                    g_inter_i.add_edges(range(n), n)
                    g_inter.append(g_inter_i)
                    mem_ids.append(th.LongTensor(segment))
                    mem_ids.append(th.LongTensor([0]))
                    n_inter += n + 1
                    readout_ids.append(n_inter - 1)

                g_inter = dgl.batch(g_inter)
                readout_ids = th.LongTensor(readout_ids)
                mem_ids = th.cat(mem_ids)

                for l, dec_layer in enumerate(self.dec_layers):
                    if len(nids_lt_pos) > 0:
                        g_dec[j].nodes[nids_lt_pos].data['k'] = g_dec[j].nodes[nids_lt_pos].data['k{}'.format(l)]
                        g_dec[j].nodes[nids_lt_pos].data['v'] = g_dec[j].nodes[nids_lt_pos].data['v{}'.format(l)]
                    g_inter.ndata['k'] = mem_key[l][mem_ids]
                    g_inter.ndata['v'] = mem_value[l][mem_ids]
                    dec_layer.infer(g_dec[j], nids_eq_pos, eids_eq_pos, nids_eq_pos_leaf,
                                    g_inter, readout_ids)
                    tmp_k[j].append(
                        g_dec[j].nodes[nids_le_pos].data['k'].view(batch_size, -1, dec_layer.h, dec_layer.d_k))
                    tmp_v[j].append(
                        g_dec[j].nodes[nids_le_pos].data['v'].view(batch_size, -1, dec_layer.h, dec_layer.d_k))

                outputs[j] = self.generator(self.cls_dropout(g_dec[j].nodes[nids_eq_pos_leaf].data['h'])).cpu()

            # clip long sentences (length >= max_len)
            for i in range(batch_size):
                for j in range(self.BEAM_SIZE):
                    if step - start_pos[i][j] >= sent_max_len:
                        tmp = outputs[j][i, eos_id].item()
                        outputs[j][i] = -float('inf')
                        outputs[j][i, eos_id] = tmp

            # update status
            logits = th.stack([outputs[j] for j in range(self.BEAM_SIZE)], 0) # (k, B, D)
            logits = logits.transpose(0, 1).contiguous() # (B, k, D)
            if step == 0:
                logits = logits[:, 0, :]
            vocab_size = logits.shape[-1]
            new_score = (th.Tensor(score).unsqueeze(-1) + logits) if step > 0 else logits
            score_t, idx_t = new_score.view(batch_size, -1).topk(self.BEAM_SIZE, -1) # (B, k)

            tmp_y = copy.deepcopy(y)
            tmp_si = copy.deepcopy(sent_indices)
            tmp_sp = copy.deepcopy(start_pos)
            tmp_eos = copy.deepcopy(eos)
            for i in range(batch_size):
                for j in range(self.BEAM_SIZE):
                    _j = idx_t[i, j].item() // vocab_size
                    token = idx_t[i, j].item() % vocab_size
                    # update status
                    score[i][j] = score_t[i, j].item()
                    # update y
                    y[i][j] = tmp_y[i][_j][:]
                    y[i][j].append(token)
                    # update g_dec
                    for l in range(len(self.dec_layers)):
                        # update k,v
                        indices = g_dec[j].filter_nodes(lambda v: (v.data['pos'] <= step),
                                                        th.arange(batch.n_nodes_per_dec_graph * i,
                                                                  batch.n_nodes_per_dec_graph * (i + 1))).cpu()
                        g_dec[j].nodes[indices].data['k{}'.format(l)] = tmp_k[_j][l][i]
                        g_dec[j].nodes[indices].data['v{}'.format(l)] = tmp_v[_j][l][i]

                    # update x
                    indices = th.arange(batch.n_nodes_per_dec_graph*i+1,
                                        batch.n_nodes_per_dec_graph*i+step+2)
                    g_dec[j].nodes[indices].data['x'] = th.LongTensor(y[i][j]).to(device)

                    # update send_indices and start_pos
                    sent_indices[i][j] = tmp_si[i][_j]
                    start_pos[i][j] = tmp_sp[i][_j]
                    eos[i][j] = tmp_eos[i][_j]
                    if token == eos_id:
                        sent_indices[i][j] += 1
                        start_pos[i][j] = step

                    if sent_indices[i][j] >= len(batch.nid_seg_arr[i]):
                        eos[i][j] = True
                        sent_indices[i][j] = len(batch.nid_seg_arr[i]) - 1

                    if j == 0 and eos[i][j]:
                        stop[i] = True

            if (stop.sum().item() == batch_size):
                break

        return [y[i][0] for i in range(batch_size)], [len(batch.nid_seg_arr[i]) for i in range(batch_size)]


def make_translate_infer_model(*args, **kwargs):
    model = SegmentTreeBeamSearch(*args, **kwargs)
    return model
