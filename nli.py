from torchtext import data, datasets
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from graph import NLIBatcher, get_nli_dataset, NLIDataset
from modules import make_matching_model
from utils import unpack_params
from optim import *
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import yaml


argparser = argparse.ArgumentParser("Natural Language Inference")
argparser.add_argument('--config', type=str)
args = argparser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f)

th.manual_seed(config['seed'])
np.random.seed(config['seed'])
th.cuda.manual_seed_all(config['seed'])

TEXT = data.Field(lower=True, tokenize='spacy', batch_first=True)
LABEL = data.LabelField(sequential=False, unk_token=None)

train, dev, test = get_nli_dataset(config['dataset']).splits(TEXT, LABEL) 
TEXT.build_vocab(train, dev, test, vectors=['glove.840B.300d'])
LABEL.build_vocab(train)

train = NLIDataset(train)
dev = NLIDataset(dev)
test = NLIDataset(test)

batcher = NLIBatcher(TEXT, LABEL, graph_type=config['graph_type'], **config.get('graph_attrs', {}))

train_loader = DataLoader(dataset=train,
                          batch_size=config['batch_size'],
                          collate_fn=batcher,
                          shuffle=True,
                          num_workers=6)
dev_loader = DataLoader(dataset=dev,
                          batch_size=config['dev_batch_size'],
                          collate_fn=batcher,
                          shuffle=False,
                          num_workers=0)
test_loader = DataLoader(dataset=test,
                          batch_size=config['batch_size'],
                          collate_fn=batcher,
                          shuffle=False,
                          num_workers=0)

dim_embed = config['dim_embed']
dim_model = config['dim_model']
dim_ff = config['dim_ff']
num_heads = config['num_heads']
n_layers = config['n_layers']
m_layers = config['m_layers']
vocab_size = len(TEXT.vocab)
n_classes = len(LABEL.vocab)

print('vocab size: {}'.format(vocab_size))

model = make_matching_model(vocab_size, dim_embed, dim_model, dim_ff, num_heads, n_classes,
                            n_layers, m_layers,
                            dropouti=config['dropouti'], dropouth=config['dropouth'],
                            dropouta=config['dropouta'], dropoutc=config['dropoutc'],
                            rel_pos=config['rel_pos'])

# load embedding
model.embed.lut.weight = nn.Parameter(TEXT.vocab.vectors)

device = th.device('cuda:0')
model = model.to(device)

embed_params, other_params, wd_params = unpack_params(model.named_parameters())

optimizer = get_wrapper(config['opt_wrapper'])(
    optim.Adam([
        {'params': embed_params, 'lr': 0},
        {'params': other_params, 'lr': config.get('lr', 1e-3)},
        {'params': wd_params, 'lr': config.get('lr', 1e-3), 'weight_decay': 5e-5}]))

best_val, test_acc = 1e9, 0

for epoch in range(config['n_epochs']):
    print('epoch {}'.format(epoch))
    print('training...')
    model.train()
    n_tokens = 0
    sum_loss = 0
    hit = 0
    for i, batch in enumerate(train_loader):
        batch.y = batch.y.to(device)
        batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
        batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
        batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)

        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = len(batch.y)
        n_tokens += n
        sum_loss += loss.item() * n
        hit += (out.max(dim=-1)[1] == batch.y).sum().item()

        if (i + 1) % config['log_interval'] == 0:
            print('loss: ', sum_loss / n_tokens, ' acc: ', hit * 1.0 / n_tokens)
            n_tokens, sum_loss, hit = 0, 0, 0

    print('evaluating...')
    model.eval()
    n_tokens = 0
    sum_loss = 0
    hit = 0
    for batch in dev_loader:
        batch.y = batch.y.to(device)
        batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
        batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
        batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)

        with th.no_grad():
            out = model(batch)
            loss = F.nll_loss(out, batch.y, reduction='sum')
            n = len(batch.y)
            n_tokens += n
            sum_loss += loss.item()
            hit += (out.max(dim=-1)[1] == batch.y).sum().item()

    print('loss: ', sum_loss / n_tokens, ' acc: ', hit * 1.0 / n_tokens)
    val_loss = sum_loss / n_tokens

    optimizer.adjust_lr()

    print('testing...')
    model.eval()
    n_tokens = 0
    sum_loss = 0
    hit = 0
    for batch in test_loader:
        batch.y = batch.y.to(device)
        batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
        batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
        batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)

        with th.no_grad():
            out = model(batch)
            loss = F.nll_loss(out, batch.y, reduction='sum')
            n = len(batch.y)
            n_tokens += n
            sum_loss += loss.item()
            hit += (out.max(dim=-1)[1] == batch.y).sum().item()

    if val_loss < best_val:
        best_val = val_loss
        test_acc = hit * 1.0 / n_tokens

    print('loss: ', sum_loss / n_tokens, ' acc: ', hit * 1.0 / n_tokens)
    print('best acc: ', test_acc)
