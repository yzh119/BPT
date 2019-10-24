import torchtext
from torchtext import data
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from graph import TCBatcher, get_text_classification_dataset, TCDataset
from modules import make_model
from utils import unpack_params
from optim import *
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
import spacy
import yaml

spacy_en = spacy.load('en')
def tokenize(x):
    return [tok.text for tok in spacy_en.tokenizer(x.replace('<br />', ' '))]

argparser = argparse.ArgumentParser("text classification")
argparser.add_argument('--config', type=str)
args = argparser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f)

random.seed(config['seed']) # fix train/dev split
th.manual_seed(config['seed'])
np.random.seed(config['seed'])
th.cuda.manual_seed_all(config['seed'])

TEXT = data.Field(lower=True, tokenize=tokenize, batch_first=True)
LABEL = data.Field(sequential=False, unk_token=None)

if config['dataset'].startswith('sst'):
    train, dev, test = get_text_classification_dataset(config['dataset']).\
        splits(TEXT, LABEL, root='./data', train_subtrees=True, fine_grained=(config['dataset'] == 'sst1'))
else:
    train, test = get_text_classification_dataset(config['dataset']).splits(TEXT, LABEL, root='./data')
    train, dev = train.split(0.9)

TEXT.build_vocab(train, dev, test, vectors=GloVe(name='840B', dim=300))
LABEL.build_vocab(train)

train = TCDataset(train, mode=config['dataset'])
dev = TCDataset(dev, mode=config['dataset'])
test = TCDataset(test, mode=config['dataset'])

batcher = TCBatcher(TEXT, LABEL, graph_type=config['graph_type'], **config.get('graph_attrs', {}))

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
                          num_workers=6)


dim_embed = config['dim_embed']
dim_model = config['dim_model']
dim_ff = config['dim_ff']
num_heads = config['num_heads']
n_layers = config['n_layers']
vocab_size = len(TEXT.vocab)
n_classes = len(LABEL.vocab)

print('vocab size: {}'.format(vocab_size))

model = make_model(vocab_size, dim_embed, dim_model, dim_ff, num_heads, n_classes, n_layers,
    dropouti=config['dropouti'], dropouth=config['dropouth'],
    dropouta=config.get('dropouta', 0.1), dropoutc=config['dropoutc'], 
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
    import time
    tic = time.time()
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

    print(time.time() - tic)

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

    print('loss: ', sum_loss / n_tokens, ' acc: ', hit * 1.0 / n_tokens, ' best acc: ', test_acc)
