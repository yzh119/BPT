from torchtext import data
from torch.utils.data import DataLoader
from graph import LMBatcher, get_lm_dataset
from graph.lm import LMDataset
from modules import make_model
from optim import get_wrapper
from utils import unpack_params
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
import time
import os

char_lm = ['enwik8', 'text8']

def run(proc_id, n_gpus, devices, config, checkpoint, eval_mode):
    th.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    th.cuda.manual_seed_all(config['seed'])

    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')

        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    TEXT = data.Field(batch_first=True)
    train, dev, test = get_lm_dataset(config['dataset']).splits(TEXT, root='./data')
    TEXT.build_vocab(train)
    train = LMDataset(train, max_length=config['length'], part=(proc_id, n_gpus))
    eval_length = config['eval_length']
    dev = LMDataset(dev, max_length=eval_length, test=True)
    test = LMDataset(test, max_length=eval_length, test=True)
    batcher = LMBatcher(TEXT,
                        graph_type=config['graph_type'],
                        **config.get('graph_attrs', {}))

    if not eval_mode:
        train_loader = DataLoader(dataset=train,
                                  batch_size=config['batch_size'] // n_gpus,
                                  collate_fn=batcher,
                                  shuffle=True,
                                  num_workers=6)

    dev_loader = DataLoader(dataset=dev,
                              batch_size=config['dev_batch_size'] // n_gpus,
                              collate_fn=batcher,
                              shuffle=False,
                              num_workers=6)
    test_loader = DataLoader(dataset=test,
                              batch_size=config['batch_size'] // n_gpus,
                              collate_fn=batcher,
                              shuffle=False,
                              num_workers=6)

    dim_embed = config['dim_embed'] 
    dim_model = config['dim_model'] 
    dim_ff = config['dim_ff']
    num_heads = config['num_heads']
    n_layers = config['n_layers']
    vocab_size = len(TEXT.vocab)
    dim_pos = config.get('dim_pos', 1)
    
    model = make_model(vocab_size, dim_embed, dim_model, dim_ff, num_heads, vocab_size, n_layers,
        dropouti=config['dropouti'], dropouth=config['dropouth'],
        dropouta=config.get('dropouta', 0.1), dropoutc=config['dropoutc'], 
        rel_pos=config['rel_pos'], dim_pos=dim_pos)

    if checkpoint != -1:
        with open('checkpoints/{}.pkl'.format(checkpoint), 'rb') as f:
            state_dict = th.load(f, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    # tie weights
    if dim_embed == dim_model:
        model.embed.lut.weight = model.generator.proj.weight

    device = th.device(dev_id)
    th.cuda.set_device(device)
    model = model.to(device)

    embed_params, other_params, wd_params = unpack_params(model.named_parameters())

    optimizer = get_wrapper(config['opt_wrapper'])(
        optim.Adam([
            {'params': embed_params + other_params, 'lr': config.get('lr', 1e-3), 'betas': (0.9, 0.98)},
            {'params': wd_params, 'lr': config.get('lr', 1e-3), 'betas': (0.9, 0.98)}]),
        **config.get('opt_attrs', {}))

    if not eval_mode:
        for _ in range(checkpoint + 1):
            for _ in range(len(train_loader)):
                optimizer.step()

    best_val = 1e9
    best_test = 0
    last_epoch = checkpoint + 2 if eval_mode else config['n_epochs']
    eval_interval = config.get('eval_interval', 1)

    for epoch in range(checkpoint + 1, last_epoch):
        if not eval_mode:
            if proc_id == 0:
                print('epoch {} starts'.format(epoch))
                print('training...')
            model.train()
            n_tokens = 0
            sum_loss = 0
            hit = 0
            tic = time.time()
            for i, batch in enumerate(train_loader):
                batch.y = batch.y.to(device)
                batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
                batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
                if dim_pos == 1:
                    batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)
                else:
                    for k in range(dim_pos):
                        batch.g.ndata['pos_{}'.format(k)] = batch.g.ndata['pos_{}'.format(k)].to(device)
                aux = (epoch * 1.0 / config['n_epochs']) if config['dataset'] in char_lm else None
                out = model(batch, aux=aux)

                if aux is None:
                    loss = F.nll_loss(out, batch.y)
                else:
                    loss = 0
                    for out_l in out:
                        loss = loss + F.nll_loss(out_l, batch.y)
                    loss /= len(out)

                optimizer.zero_grad()
                loss.backward()
                if n_gpus > 1:
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            th.distributed.all_reduce(param.grad.data,
                                                      op=th.distributed.ReduceOp.SUM)
                            param.grad.data /= n_gpus

                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                n = len(batch.y)
                n_tokens += n
                sum_loss += loss.item() * n
                if aux is None:
                    hit += (out.max(dim=-1)[1] == batch.y).sum().item()
                else:
                    hit += (out[-1].max(dim=-1)[1] == batch.y).sum().item()

                if (i + 1) % config['log_interval'] == 0 and proc_id == 0:
                    mem = th.cuda.max_memory_cached()
                    print('ppl: ', np.exp(sum_loss / n_tokens), ' acc: ', hit * 1.0 / n_tokens,
                          ' #tokens/s: ', config['batch_size'] * config['log_interval'] * config['length'] / (time.time() - tic),
                          ' #mem: ', mem / 1024 / 1024 / 1024)
                    tic = time.time()
                    n_tokens, sum_loss, hit = 0, 0, 0
        
        if n_gpus > 1:
            th.distributed.barrier()
        if proc_id == 0:
            print('evaluating...')

            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            with open('checkpoints/{}.pkl'.format(epoch), 'wb') as f:
                th.save(model.state_dict(), f)
        
        if (epoch + 1) % eval_interval > 0 and not eval_mode:
            continue

        model.eval()
        n_tokens = 0
        sum_loss = 0
        hit = 0
        for batch in dev_loader:
            batch.y = batch.y.to(device)
            batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
            batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
            if dim_pos == 1:
                batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)
            else:
                for k in range(dim_pos):
                    batch.g.ndata['pos_{}'.format(k)] = batch.g.ndata['pos_{}'.format(k)].to(device)

            with th.no_grad():
                out = model(batch)
                loss = F.nll_loss(out, batch.y, reduction='sum')
                n = len(batch.y)
                n_tokens += n
                sum_loss += loss.item()
                hit += (out.max(dim=-1)[1] == batch.y).sum().item()

        if proc_id == 0:
            if config['dataset'] in char_lm:
                print('bpc: ', (sum_loss / n_tokens) / np.log(2), ' acc: ', hit * 1.0 / n_tokens)
            else:
                print('ppl: ', np.exp(sum_loss / n_tokens), ' acc: ', hit * 1.0 / n_tokens)
        optimizer.adjust_lr(np.exp(sum_loss / n_tokens))
        val_ppl = np.exp(sum_loss / n_tokens)

        if proc_id == 0:
            print('testing...')
        model.eval()
        n_tokens = 0
        sum_loss = 0
        hit = 0
        for batch in test_loader:
            batch.y = batch.y.to(device)
            batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
            batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
            if dim_pos == 1:
                batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)
            else:
                for k in range(dim_pos):
                    batch.g.ndata['pos_{}'.format(k)] = batch.g.ndata['pos_{}'.format(k)].to(device)

            with th.no_grad():
                out = model(batch)
                loss = F.nll_loss(out, batch.y, reduction='sum')
                n = len(batch.y)
                n_tokens += n
                sum_loss += loss.item()
                hit += (out.max(dim=-1)[1] == batch.y).sum().item()

        if proc_id == 0:
            if config['dataset'] in char_lm:
                print('bpc: ', (sum_loss / n_tokens) / np.log(2), ' acc: ', hit * 1.0 / n_tokens)
            else:
                print('ppl: ', np.exp(sum_loss / n_tokens), ' acc: ', hit * 1.0 / n_tokens)

        if val_ppl < best_val:
            best_val = val_ppl
            best_test = np.exp(sum_loss / n_tokens)

        if proc_id == 0:
            if config['dataset'] in char_lm:
                print('best val: %.2f ' % np.log2(best_val), 'best test: %.2f ' % np.log2(best_test))
            else:
                print('best val: %.2f ' % best_val, 'best test: %.2f ' % best_test)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("language modeling")
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--gpu', type=str, default='0')
    argparser.add_argument('--eval', action='store_true')
    argparser.add_argument('--checkpoint', type=int, default=-1)
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    devices = list(map(int, args.gpu.split(',')))

    n_gpus = len(devices)
    if n_gpus == 1:
        run(0, n_gpus, devices, config, args.checkpoint, args.eval)
    else:
        mp = th.multiprocessing
        mp.spawn(run, args=(n_gpus, devices, config, args.checkpoint, args.eval), nprocs=n_gpus)
