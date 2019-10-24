from torchtext import data
from torch.utils.data import DataLoader
from graph import MTInferBatcher, get_mt_dataset, MTDataset, DocumentMTDataset
from modules import make_translate_infer_model
from utils import tensor_to_sequence, average_model

import torch as th
import argparse
import yaml

max_length = 1024

def run(dev_id, config):
    _dataset = config['dataset']

    if _dataset == 'iwslt':
        TEXT = [data.Field(batch_first=True) for _ in range(2)]
        dataset = get_mt_dataset('iwslt')
        _, _, test = dataset.splits(exts=('.tc.zh', '.tc.en'), fields=TEXT, root='./data')
        test = DocumentMTDataset(test, context_length=config['context_len'])
        vocab_zh, vocab_en = dataset.load_vocab(root='./data')
        print('vocab size: ', len(vocab_zh), len(vocab_en))
        vocab_sizes = [len(vocab_zh), len(vocab_en)]
        TEXT[0].vocab = vocab_zh
        TEXT[1].vocab = vocab_en
        batcher = MTInferBatcher(TEXT, config['doc_max_len'], test.BOS_TOKEN,
                                 graph_type=config['graph_type'], **config.get('graph_attrs', {}))
        test_loader = DataLoader(dataset=test,
                                 batch_size=config['test_batch_size'],
                                 collate_fn=batcher,
                                 shuffle=False)

    elif _dataset == 'wmt':
        TEXT = data.Field(batch_first=True)
        dataset = get_mt_dataset('wmt14')
        _, _, test = dataset.splits(exts=['.en', '.de'], fields=[TEXT, TEXT], root='./data')
        test = MTDataset(test)
        vocab = dataset.load_vocab(root='./data')[0]
        print('vocab size: ', len(vocab))
        vocab_sizes = [len(vocab)]
        TEXT.vocab = vocab
        batcher = MTInferBatcher(TEXT, config['doc_max_len'], test.BOS_TOKEN,
                                 graph_type=config['graph_type'], **config.get('graph_attrs', {}))
        test_loader = DataLoader(dataset=test,
                                 batch_size=config['test_batch_size'],
                                 collate_fn=batcher,
                                 shuffle=False)
    elif _dataset == 'multi':
        TEXT = [data.Field(batch_first=True) for _ in range(2)]
        dataset = get_mt_dataset('multi30k')
        _, _, test = dataset.splits(exts=['.en.atok', '.de.atok'], fields=TEXT, root='./data')
        test = MTDataset(test)
        vocab_en, vocab_de = dataset.load_vocab(root='./data')
        print('vocab size: ', len(vocab_en), len(vocab_de))
        vocab_sizes = [len(vocab_en), len(vocab_de)]
        TEXT[0].vocab = vocab_en
        TEXT[1].vocab = vocab_de
        batcher = MTInferBatcher(TEXT, config['doc_max_len'], test.BOS_TOKEN,
                                 graph_type=config['graph_type'], **config.get('graph_attrs', {}))
        test_loader = DataLoader(dataset=test,
                                 batch_size=config['test_batch_size'],
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

    model = make_translate_infer_model(vocab_sizes, dim_model, dim_ff, num_heads,
                                       n_layers, m_layers,
                                       dropouti=dropouti, dropouth=dropouth,
                                       dropouta=dropouta, dropoutc=dropoutc,
                                       rel_pos=rel_pos)

    device = th.device(dev_id)
    model.load_state_dict(
        average_model(['{}-{}.pkl'.format(epoch, config['save_name']) for epoch in range(config['n_epochs'] - 5, config['n_epochs'])]))
    model = model.to(device)

    model.eval()

    if _dataset == 'iwslt':
        vocab_trg = vocab_en
    elif _dataset == 'wmt':
        vocab_trg = vocab
    elif _dataset == 'multi':
        vocab_trg = vocab_de

    for batch in test_loader:
        with th.no_grad():
            batch.g_enc.edata['etype'] = batch.g_enc.edata['etype'].to(device)
            batch.g_enc.ndata['pos'] = batch.g_enc.ndata['pos'].to(device)
            batch.g_enc.ndata['x'] = batch.g_enc.ndata['x'].to(device)
            for j in range(batcher.k):
                batch.g_dec[j].edata['etype'] = batch.g_dec[j].edata['etype'].to(device)
                batch.g_dec[j].ndata['pos'] = batch.g_dec[j].ndata['pos'].to(device)
                batch.g_dec[j].ndata['x'] = batch.g_dec[j].ndata['x'].to(device)
            output = model(batch, vocab_trg.stoi[MTDataset.EOS_TOKEN], sent_max_len=config['sent_max_len'])
        for sequence in tensor_to_sequence(vocab_trg.itos, output, batch.n_sent_ctx):
            print(sequence)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("machine translation inference")
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--gpu', type=int, default=0)
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    run(args.gpu, config)
