from torchtext import data
from torch.utils.data import DataLoader
from graph import MTBatcher, get_mt_dataset, MTDataset, DocumentMTDataset
from modules import make_translation_model
from optim import get_wrapper
from loss import LabelSmoothing

import numpy as np
import torch as th
import torch.optim as optim
import argparse
import yaml
import os


def run(proc_id, n_gpus, devices, config, checkpoint):
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

    _dataset = config['dataset']
    grad_accum = config['grad_accum']

    if _dataset == 'iwslt':
        TEXT = [data.Field(batch_first=True) for _ in range(2)]
        dataset = get_mt_dataset('iwslt')
        train, dev, test = dataset.splits(exts=('.tc.zh', '.tc.en'), fields=TEXT, root='./data')
        train = DocumentMTDataset(train, context_length=config['context_len'], part=(proc_id, n_gpus))
        dev = DocumentMTDataset(dev, context_length=config['context_len'])
        test = DocumentMTDataset(test, context_length=config['context_len'])
        vocab_zh, vocab_en = dataset.load_vocab(root='./data')
        print('vocab size: ', len(vocab_zh), len(vocab_en))
        vocab_sizes = [len(vocab_zh), len(vocab_en)]
        TEXT[0].vocab = vocab_zh
        TEXT[1].vocab = vocab_en
        batcher = MTBatcher(TEXT, graph_type=config['graph_type'], **config.get('graph_attrs', {}))
        train_loader = DataLoader(dataset=train,
                                  batch_size=config['batch_size'] // n_gpus,
                                  collate_fn=batcher,
                                  shuffle=True,
                                  num_workers=6)
        dev_loader = DataLoader(dataset=dev,
                                batch_size=config['dev_batch_size'],
                                collate_fn=batcher,
                                shuffle=False)
        test_loader = DataLoader(dataset=test,
                                 batch_size=config['dev_batch_size'],
                                 collate_fn=batcher,
                                 shuffle=False)

    elif _dataset == 'wmt':
        TEXT = data.Field(batch_first=True)
        dataset = get_mt_dataset('wmt14')
        train, dev, test = dataset.splits(exts=['.en', '.de'], fields=[TEXT, TEXT], root='./data')
        train = MTDataset(train, part=(proc_id, n_gpus))
        dev = MTDataset(dev)
        test = MTDataset(test)
        vocab = dataset.load_vocab(root='./data')[0]
        print('vocab size: ', len(vocab))
        vocab_sizes = [len(vocab)]
        TEXT.vocab = vocab
        batcher = MTBatcher(TEXT, graph_type=config['graph_type'], **config.get('graph_attrs', {}))
        train_loader = DataLoader(dataset=train,
                                  batch_size=config['batch_size'] // n_gpus,
                                  collate_fn=batcher,
                                  shuffle=True,
                                  num_workers=6)
        dev_loader = DataLoader(dataset=dev,
                                batch_size=config['dev_batch_size'],
                                collate_fn=batcher,
                                shuffle=False)
        test_loader = DataLoader(dataset=test,
                                 batch_size=config['dev_batch_size'],
                                 collate_fn=batcher,
                                 shuffle=False)
    elif _dataset == 'multi':
        TEXT = [data.Field(batch_first=True) for _ in range(2)]
        dataset = get_mt_dataset('multi30k')
        train, dev, test = dataset.splits(exts=['.en.atok', '.de.atok'], fields=TEXT, root='./data')
        train = MTDataset(train, part=(proc_id, n_gpus))
        dev = MTDataset(dev)
        test = MTDataset(test)
        vocab_en, vocab_de = dataset.load_vocab(root='./data')
        print('vocab size: ', len(vocab_en), len(vocab_de))
        vocab_sizes = [len(vocab_en), len(vocab_de)]
        TEXT[0].vocab = vocab_en
        TEXT[1].vocab = vocab_de
        batcher = MTBatcher(TEXT, graph_type=config['graph_type'], **config.get('graph_attrs', {}))
        train_loader = DataLoader(dataset=train,
                                  batch_size=config['batch_size'] // n_gpus,
                                  collate_fn=batcher,
                                  shuffle=True,
                                  num_workers=6)
        dev_loader = DataLoader(dataset=dev,
                                batch_size=config['dev_batch_size'],
                                collate_fn=batcher,
                                shuffle=False)
        test_loader = DataLoader(dataset=test,
                                 batch_size=config['dev_batch_size'],
                                 collate_fn=batcher,
                                 shuffle=False)


    dim_model = config['dim_model']
    dim_ff = config['dim_ff']
    num_heads = config['num_heads']
    n_layers = config['n_layers']
    m_layers = config['m_layers']
    dropouti = config['dropouti']
    dropouth = config['dropouth']
    dropouta = config['dropouta']
    dropoutc = config['dropoutc']
    rel_pos = config['rel_pos']

    model = make_translation_model(vocab_sizes, dim_model, dim_ff, num_heads,
                                   n_layers, m_layers,
                                   dropouti=dropouti, dropouth=dropouth,
                                   dropouta=dropouta, dropoutc=dropoutc,
                                   rel_pos=rel_pos)

    if checkpoint != -1:
        with open('checkpoints/{}-{}.pkl'.format(checkpoint, config['save_name']), 'rb') as f:
            state_dict = th.load(f, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    # tie weight
    if config.get('share_weight', False):
        model.embed[-1].lut.weight = model.generator.proj.weight

    criterion = LabelSmoothing(vocab_sizes[-1], smoothing=0.1)

    device = th.device(dev_id)
    th.cuda.set_device(device)
    model, criterion = model.to(device), criterion.to(device)

    n_epochs = config['n_epochs']
    optimizer = get_wrapper('noam')(
        dim_model, config['factor'], config.get('warmup', 4000),
        optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9,
                   weight_decay=config.get('weight_decay', 0)))

    for _ in range(checkpoint + 1):
        for _ in range(len(train_loader)):
            optimizer.step()

    log_interval = config['log_interval']

    for epoch in range(checkpoint + 1, n_epochs):
        if proc_id == 0:
            print("epoch {}".format(epoch))
            print("training...")
        model.train()

        tot = 0
        hit = 0
        loss_accum = 0
        for i, batch in enumerate(train_loader):
            batch.y = batch.y.to(device)
            batch.g_enc.edata['etype'] = batch.g_enc.edata['etype'].to(device)
            batch.g_enc.ndata['x'] = batch.g_enc.ndata['x'].to(device)
            batch.g_enc.ndata['pos'] = batch.g_enc.ndata['pos'].to(device)
            batch.g_dec.edata['etype'] = batch.g_dec.edata['etype'].to(device)
            batch.g_dec.ndata['x'] = batch.g_dec.ndata['x'].to(device)
            batch.g_dec.ndata['pos'] = batch.g_dec.ndata['pos'].to(device)
            out = model(batch)
            loss = criterion(out, batch.y) / len(batch.y)
            loss_accum += loss.item() * len(batch.y)
            tot += len(batch.y)
            hit += (out.max(dim=-1)[1] == batch.y).sum().item()
            if proc_id == 0:
                if (i + 1) % log_interval == 0:
                    print('step {}, loss : {}, acc : {}'.format(i, loss_accum / tot, hit / tot))
                    tot = 0
                    hit = 0
                    loss_accum = 0
            loss.backward()

            if (i + 1) % grad_accum == 0:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        if n_gpus > 1:
                            th.distributed.all_reduce(param.grad.data,
                                                      op=th.distributed.ReduceOp.SUM)
                            param.grad.data /= (n_gpus * grad_accum)
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        tot = 0
        hit = 0
        loss_accum = 0
        for batch in dev_loader:
            with th.no_grad():
                batch.y = batch.y.to(device)
                batch.g_enc.edata['etype'] = batch.g_enc.edata['etype'].to(device)
                batch.g_enc.ndata['x'] = batch.g_enc.ndata['x'].to(device)
                batch.g_enc.ndata['pos'] = batch.g_enc.ndata['pos'].to(device)
                batch.g_dec.edata['etype'] = batch.g_dec.edata['etype'].to(device)
                batch.g_dec.ndata['x'] = batch.g_dec.ndata['x'].to(device)
                batch.g_dec.ndata['pos'] = batch.g_dec.ndata['pos'].to(device)
                out = model(batch)
                loss_accum += criterion(out, batch.y)
                tot += len(batch.y)
                hit += (out.max(dim=-1)[1] == batch.y).sum().item()

        if n_gpus > 1:
            th.distributed.barrier()
        if proc_id == 0:
            print('evaluate...')
            print('loss : {}, acc : {}'.format(loss_accum / tot, hit / tot))

        tot = 0
        hit = 0
        loss_accum = 0
        for batch in test_loader:
            with th.no_grad():
                batch.y = batch.y.to(device)
                batch.g_enc.edata['etype'] = batch.g_enc.edata['etype'].to(device)
                batch.g_enc.ndata['x'] = batch.g_enc.ndata['x'].to(device)
                batch.g_enc.ndata['pos'] = batch.g_enc.ndata['pos'].to(device)
                batch.g_dec.edata['etype'] = batch.g_dec.edata['etype'].to(device)
                batch.g_dec.ndata['x'] = batch.g_dec.ndata['x'].to(device)
                batch.g_dec.ndata['pos'] = batch.g_dec.ndata['pos'].to(device)
                out = model(batch)
                loss_accum += criterion(out, batch.y)
                tot += len(batch.y)
                hit += (out.max(dim=-1)[1] == batch.y).sum().item()

        if n_gpus > 1:
            th.distributed.barrier()
        if proc_id == 0:
            print('testing...')
            print('loss : {}, acc : {}'.format(loss_accum / tot, hit / tot))

            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')   
            with open('checkpoints/{}-{}.pkl'.format(epoch, config['save_name']), 'wb') as f:
                th.save(model.state_dict(), f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("machine translation")
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--gpu', type=str, default='0')
    argparser.add_argument('--checkpoint', type=int, default=-1)
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    devices = list(map(int, args.gpu.split(',')))

    n_gpus = len(devices)
    if n_gpus == 1:
        run(0, n_gpus, devices, config, args.checkpoint)
    else:
        mp = th.multiprocessing
        mp.spawn(run, args=(n_gpus, devices, config, args.checkpoint), nprocs=n_gpus)
