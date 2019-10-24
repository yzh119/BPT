import torch as th

def unpack_params(params):
    embed_params, other_params, wd_params = [], [], []
    for k, v in params:
        if 'embed' in k:
            embed_params.append(v)
        elif 'norm' in k or 'bias' in k:
            other_params.append(v)
        else: # applies weight decay
            wd_params.append(v)

    return embed_params, other_params, wd_params

def tensor_to_sequence(itos, output, n_sent_ctx):
    tensor, n_sentences = output
    rst = []
    for row, n_sentence, n_sent_ctx_i in zip(tensor, n_sentences, n_sent_ctx):
        cnt = 0
        sent_i = []
        for token in row:
            if itos[token] == '<eos>':
                if cnt >= n_sent_ctx_i:
                    sent_i.append('\n')
                cnt += 1
                if cnt == n_sentence:
                    break
            elif cnt >= n_sent_ctx_i:
                sent_i.append(itos[token] + ' ')
        rst.append(''.join(sent_i).strip())
    return rst

def average_model(model_files):
    ret = {}
    for i, model_name in enumerate(model_files):
        with open('checkpoints/{}'.format(model_name), 'rb') as f:
            state_dict = th.load(f, map_location=lambda storage, loc: storage)
        if len(ret) == 0:
            ret = state_dict
        else:
            for k in ret.keys():
                ret[k].mul_(i).add_(state_dict[k]).div_(i + 1)
    return ret
