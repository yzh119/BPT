from .attention import STTBlock, STTMemBlock
from .embedding import PositionalEncoding, Embedding, KDEmbedding

import numpy as np
import torch as th
import torch.nn as nn

class Generator(nn.Module):
    """
    log(softmax(Wx + b))
    """
    def __init__(self, dim_model, n_classes):
        super(Generator, self).__init__()
        self.proj = nn.Linear(dim_model, n_classes)

    def forward(self, x):
        return th.log_softmax(self.proj(x), dim=-1)

def move_to_device(_tuple, device):
    return tuple(x.to(device) if isinstance(x, th.Tensor) else move_to_device(x, device) for x in _tuple)

class SegmentTreeEncoderDecoder(nn.Module):
    def __init__(self, vocab_sizes, dim_model, dim_ff, h,
                 n_layers, m_layers,
                 dropouti=0.1, dropouth=0.1, dropouta=0.1, dropoutc=0.1, rel_pos=False):
        super(SegmentTreeEncoderDecoder, self).__init__()
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.h = h
        assert self.dim_model % self.h == 0
        self.rel_pos = rel_pos
        self.vocab_sizes = vocab_sizes
        self.n_classes = vocab_sizes[-1]

        self.embed = nn.ModuleList([
            Embedding(vocab_size, dim_model)
            for vocab_size in vocab_sizes
        ])

        if not self.rel_pos:
            self.pos_enc = PositionalEncoding(dim_model)

        self.emb_dropout = nn.Dropout(dropouti)
        self.cls_dropout = nn.Dropout(dropoutc)

        enc_layer_list = []
        for _ in range(n_layers):
            enc_layer_list.append(
                STTBlock(self.dim_model, self.h, self.dim_ff,
                         rel_pos=self.rel_pos, drop_h=dropouth, drop_a=dropouta)
            )
        self.enc_layers = nn.ModuleList(enc_layer_list)

        dec_layer_list = []
        for _ in range(m_layers):
            dec_layer_list.append(
                STTMemBlock(self.dim_model, self.h, self.dim_ff,
                            rel_pos=self.rel_pos, drop_h=dropouth, drop_a=dropouta)
            )
        self.dec_layers = nn.ModuleList(dec_layer_list)
        self.generator = Generator(dim_model, self.n_classes)
        self.norm = nn.LayerNorm(dim_model)

    def reset_parameters(self):
        for layer in self.enc_layers:
            layer.reset_parameters()
        for layer in self.dec_layers:
            layer.reset_parameters()

    def forward(self, batch):
        g_enc = batch.g_enc.local_var()
        g_dec = batch.g_dec.local_var()
        g_inter = batch.g_inter.local_var()
        device = next(self.parameters()).device
        csr_enc = move_to_device(batch.g_enc_csr, device)
        csr_dec = move_to_device(batch.g_dec_csr, device)
        csr_inter = move_to_device(batch.g_inter_csr, device)

        leaf_ids = batch.leaf_ids['enc']
        h_enc = self.embed[0](g_enc.nodes[leaf_ids].data['x'])
        if not self.rel_pos:
            h_enc += self.pos_enc(g_enc.nodes[leaf_ids].data['pos'])
        g_enc.nodes[leaf_ids].data['h'] = self.norm(self.emb_dropout(h_enc))

        for enc_layer in self.enc_layers:
            enc_layer(g_enc, csr_enc)

        mem = g_enc.ndata['h']
        # this part of code would be deprecated after using bipartite graph.
        if g_enc.number_of_nodes() < g_inter.number_of_nodes():
            mem = th.cat([mem, mem.new_zeros(
                g_inter.number_of_nodes() - g_enc.number_of_nodes(),
                *mem.shape[1:])], 0)

        leaf_ids = batch.leaf_ids['dec']
        h_dec = self.embed[-1](g_dec.nodes[leaf_ids].data['x'])
        if not self.rel_pos:
            h_dec += self.pos_enc(g_dec.nodes[leaf_ids].data['pos'])
        g_dec.nodes[leaf_ids].data['h'] = self.norm(self.emb_dropout(h_dec))

        for dec_layer in self.dec_layers:
            dec_layer(g_dec, csr_dec, mem, csr_inter)

        output = self.generator(self.cls_dropout(g_dec.nodes[batch.readout_ids].data['h']))

        return output

class SegmentTreeMatching(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_model, dim_ff, h, n_classes,
                 n_layers, m_layers,
                 dropouti=0.1, dropouth=0.1, dropouta=0.1, dropoutc=0, rel_pos=False):
        super(SegmentTreeMatching, self).__init__()
        self.dim_embed = dim_embed
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.h = h
        assert self.dim_model % self.h == 0
        self.rel_pos = rel_pos

        self.embed = Embedding(vocab_size, dim_embed)
        if dim_embed != dim_model:
            self.embed_to_hidden = nn.Linear(dim_embed, dim_model)
        else:
            self.embed_to_hidden = nn.Identity()

        if not self.rel_pos:
            self.pos_enc = PositionalEncoding(dim_embed)

        self.emb_dropout = nn.Dropout(dropouti)
        self.cls_dropout = nn.Dropout(dropoutc)

        enc_layer_list = []
        for _ in range(n_layers):
            enc_layer_list.append(
                STTBlock(self.dim_model, self.h, self.dim_ff,
                         rel_pos=self.rel_pos, drop_h=dropouth, drop_a=dropouta))
        self.enc_layers = nn.ModuleList(enc_layer_list)

        dec_layer_list = []
        for _ in range(m_layers):
            dec_layer_list.append(
                STTMemBlock(self.dim_model, self.h, self.dim_ff,
                            rel_pos=self.rel_pos, drop_h=dropouth, drop_a=dropouta)
            )
        self.dec_layers = nn.ModuleList(dec_layer_list)

        self.generator = nn.Sequential(
            nn.Linear(2 * dim_model, dim_model),
            nn.Dropout(dropoutc),
            nn.ReLU(),
            nn.Linear(dim_model, n_classes),
            nn.LogSoftmax(dim=-1)
        )
        self.norm = nn.LayerNorm(dim_model)

    def reset_parameters(self):
        for layer in self.enc_layers:
            layer.reset_parameters()
        for layer in self.dec_layers:
            layer.reset_parameters()

    def forward(self, batch):
        g = batch.g.local_var()
        leaf_ids = batch.leaf_ids
        device = next(self.parameters()).device
        csr = move_to_device(batch.g_csr, device)

        # get embedding
        h = self.embed(g.nodes[leaf_ids].data['x'])

        # embed to hidden
        g.nodes[leaf_ids].data['h'] = self.embed_to_hidden(h)

        # add pos encoding
        if not self.rel_pos:
            g.nodes[leaf_ids].data['h'] += self.pos_enc(g.nodes[leaf_ids].data['pos'])

        # input dropout
        g.nodes[leaf_ids].data['h'] = self.norm(self.emb_dropout(g.nodes[leaf_ids].data['h']))

        # go through the layers
        for layer in self.enc_layers:
            layer(g, csr)

        g_inter = batch.g_inter.local_var()
        mem = g.ndata['h']
        csr_inter = move_to_device(batch.g_inter_csr, device)

        for layer in self.dec_layers:
            layer(g, csr, mem, csr_inter)

        h = g.nodes[batch.readout_ids].data['h'].view(-1, 2 * self.dim_model)
        output = self.generator(self.cls_dropout(h))

        return output

class SegmentTreeTransformer(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_model, dim_ff, h, n_classes, n_layers,
                 dropouti=0.1, dropouth=0.1, dropouta=0.1, dropoutc=0, rel_pos=False,
                 dim_pos=1):
        super(SegmentTreeTransformer, self).__init__()
        self.dim_embed = dim_embed
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.h = h
        assert self.dim_model % self.h == 0
        self.rel_pos = rel_pos

        self.embed = Embedding(vocab_size, dim_embed)

        if dim_embed != dim_model:
            self.embed_to_hidden = nn.Linear(dim_embed, dim_model)
        else:
            self.embed_to_hidden = nn.Identity()

        if not self.rel_pos:
            if dim_pos == 1:
                self.pos_enc = PositionalEncoding(dim_model)
            else:
                self.pos_enc = KDEmbedding(dim_model, dim_pos)

        self.dim_pos = dim_pos
        self.emb_dropout = nn.Dropout(dropouti)
        self.cls_dropout = nn.Dropout(dropoutc)

        layer_list = []
        for _ in range(n_layers):
            layer_list.append(
                STTBlock(self.dim_model, self.h, self.dim_ff,
                         rel_pos=self.rel_pos, drop_h=dropouth, drop_a=dropouta))
        self.layers = nn.ModuleList(layer_list)

        self.generator = Generator(dim_model, n_classes)
        self.norm = nn.LayerNorm(dim_model)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batch, aux=None):
        g = batch.g.local_var()
        leaf_ids = batch.leaf_ids
        device = next(self.parameters()).device
        csr = move_to_device(batch.g_csr, device)

        # get embedding
        h = self.embed(g.nodes[leaf_ids].data['x'])

        # embed to hidden
        g.nodes[leaf_ids].data['h'] = self.embed_to_hidden(h)

        # add pos encoding
        if not self.rel_pos:
            if self.dim_pos > 1:
                g.nodes[leaf_ids].data['h'] += self.pos_enc(
                    *(g.nodes[leaf_ids].data['pos_{}'.format(dim)] for dim in range(self.dim_pos))
                )
            else:
                g.nodes[leaf_ids].data['h'] += self.pos_enc(g.nodes[leaf_ids].data['pos'])

        # input dropout
        g.nodes[leaf_ids].data['h'] = self.norm(self.emb_dropout(g.nodes[leaf_ids].data['h']))

        if aux is not None:
            outputs = []

        # go through the layers
        for i, layer in enumerate(self.layers):
            layer(g, csr)
            if aux is not None:
                if (i + 1) > len(self.layers) * aux:
                    outputs.append(self.generator(self.cls_dropout(g.nodes[batch.readout_ids].data['h'])))

        # output
        output = self.generator(self.cls_dropout(g.nodes[batch.readout_ids].data['h']))

        if aux is not None:
            return outputs
        else:
            return output

def make_model(*args, **kwargs):
    model = SegmentTreeTransformer(*args, **kwargs)
    model.reset_parameters()
    return model

def make_matching_model(*args, **kwargs):
    model = SegmentTreeMatching(*args, **kwargs)
    model.reset_parameters()
    return model

def make_translation_model(*args, **kwargs):
    model = SegmentTreeEncoderDecoder(*args, **kwargs)
    model.reset_parameters()
    return model
