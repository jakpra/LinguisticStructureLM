import sys
import random
from collections import namedtuple, defaultdict, Counter

from network2tikz import plot

import torch

from util import get_node_text, compute_structure_stats, cos_sim, mtl_result


_UPOS_CLASSES = {'f_adp': {'ADP'},
                 'f_part': {'PART'},
                 'f_cconj': {'CCONJ',},
                 'f_sconj': {'SCONJ'},
                 'f_pron': {'PRON'},
                 'f_det': {'DET'},
                 'f_aux': {'AUX'},
                 'punctuation': {'PUNCT'},
                 'number': {'NUM'},
                 'c_noun': {'NOUN', 'PROPN'},
                 'c_verb': {'VERB'},
                 'c_mod': {'ADJ', 'ADV'},
                 'c_misc': {'INTJ', 'SYM', 'X'}
                }
# _UPOS_CLASSES = {'function': {'ADP', 'PART', 'CCONJ', 'SCONJ', 'PRON', 'DET', 'AUX'},
#                  'punctuation': {'PUNCT'},
#                  'number': {'NUM'},
#                  'content': {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'INTJ', 'SYM', 'X'}
#                 }
UPOS_CLASSES = {}
for k, v in _UPOS_CLASSES.items():
    for k2 in v:
        UPOS_CLASSES[k2] = k

AUX_LOSSES = ['aux loss', 'hi_res loss', 'lo_res loss', 'emb loss']


def calib(confs, accs):
    return torch.mean(torch.abs(torch.tensor(confs) - torch.tensor(accs)))


def get_interesting_tokens(logits1, logits2, labels, n=None):
    logits1 = torch.log_softmax(logits1, dim=-1).detach()
    logits2 = torch.log_softmax(logits2, dim=-1).detach()

    nll1 = torch.nn.functional.nll_loss(logits1, labels, reduction='none')
    nll2 = torch.nn.functional.nll_loss(logits2, labels, reduction='none')
    nll_diff = nll2 - nll1

    sorted_nlls, sorted_nll_idxs = torch.sort(nll_diff)
    sorted_nlls = sorted_nlls.view(-1)
    sorted_nll_idxs = sorted_nll_idxs.view(-1)

    if n is not None and sorted_nll_idxs.size(0) > 2*n:
        return torch.cat([sorted_nll_idxs[:n], sorted_nll_idxs[-n:]], dim=0), torch.cat([sorted_nlls[:n], sorted_nlls[-n:]], dim=0)
    return sorted_nll_idxs, sorted_nlls


# algorithm from: On Some Pitfalls in Automatic Evaluation and Significance Testing for MT (Riezler & Maxwell, 2005)
def approx_rand_significance(outputs_a, outputs_b, output_sizes, R=10000, alphas=(0.05, 0.01, 0.005, 0.001),
                             aggregate='mean'):
    n = len(output_sizes)
    assert n == len(outputs_b) == len(outputs_a)
    n_tok = sum(output_sizes)

    if aggregate == 'mean':
        aggr_fn = lambda s: sum(s) / n_tok
    elif aggregate == 'exp_mean':
        aggr_fn = lambda s: torch.exp(sum(s) / n_tok).item()
    else:
        raise NotImplementedError

    real_diff = abs(aggr_fn(outputs_a) - aggr_fn(outputs_b))

    c = 0
    random_change = {0: real_diff}
    idxs = list(range(n))
    for r in range(R):
        random.shuffle(idxs)
        shuffle_a, shuffle_b = [], []
        for i in idxs:
            if random.randint(0, 1) == 1:
                shuffle_a.append(outputs_b[i])
                shuffle_b.append(outputs_a[i])
            else:
                shuffle_a.append(outputs_a[i])
                shuffle_b.append(outputs_b[i])
        pseudo_diff = abs(aggr_fn(shuffle_a) - aggr_fn(shuffle_b))
        if pseudo_diff >= real_diff:
            c += 1
            random_change[r+1] = pseudo_diff
    p = (c + 1) / (R + 1)
    return p, random_change


def anchor_overlap(a, start, end):
    a_start = a['from']
    a_end = a['to']
    return (a_start <= start and a_end > start) or \
           (a_end >= end and a_start < end) or \
           (a_start >= start and a_end <= end)


def get_all_structure_stats(sites, data, edge_labels, tokenizer, verbose=False):
    sites_dict = defaultdict(list)
    for d, s, t in sites:
        sites_dict[s].append((d, t))
    graphs_dict = {x['id']: x for x in data if x['id'] in sites_dict}
    special_len = len('<|endoftext|>')

    rels = ['parent', 'siblings', 'grandparents', 'aunts', 'child', 'coparents']
    n_rels = defaultdict(list)
    n_rel_labels = defaultdict(lambda: defaultdict(list))
    n_una = 0

    for d, s, t in sites:
        graph = graphs_dict[s]
        id2node = graph['id2node']
        node2text = graph['node2text']
        chars2node = graph['chars2node']
        text = graph['text']
        all_target_idxs = tokenizer(f'<|endoftext|>{text}', return_tensors='pt')
        token_idxs = all_target_idxs.input_ids[0, 1:]
        tokens = all_target_idxs.encodings[0].tokens[1:]
        marker = ' ' * (len(text) + 1)
        try:
            char, end_char = all_target_idxs.token_to_chars(t+1)
            char -= special_len
            end_char -= special_len
        except TypeError:
            t = len(token_idxs)
            char, end_char = all_target_idxs.token_to_chars(t)
            char -= special_len
            end_char -= special_len
        marker = marker[:char+1] + '^' * (end_char - char) + marker[end_char+1:]

        node_ids = [x for x, n in id2node.items() if 'anchors' in n and any(anchor_overlap(a, char, end_char) for a in n['anchors'])]
        # TODO: any(anchor_overlap(a, char, end_char) ...) should (?) be equivalent to util.get_offset_diff([char, end_char], x, id2node)[1] == 0
        node_ids = (((len(id2node[x].get('parents', [])),
                      len(id2node[x].get('children', []))), x) for x in node_ids)
        _node_ids = sorted([(x, y) for x, y in node_ids],
                           reverse=True)

        try:
            node_id = _node_ids[0][1]
            anchor = get_node_text(node_id, node2text, id2node)
        except IndexError:
            node_id = anchor = None

        try:
            tok_idx = token_idxs[t]
            tok = tokenizer.decode(tok_idx)
        except IndexError:
            tok = tok_idx = None

        if verbose:
            print(marker, d, t, tok_idx, char, tok, node_id, anchor)

        if node_id is not None:
            local_n_rels, local_n_rel_labels = compute_structure_stats(node_id, id2node, node2text)
            for r in rels:
                n_rels[r].append(local_n_rels[r])
                for l in edge_labels:
                    n_rel_labels[r][l].append(local_n_rel_labels[r][l])

            if tok is not None and len(tok.strip()) < len(anchor):
                n_una += 1

    n = len(list(n_rels.values())[0])

    rel_stats = []
    label_stats = defaultdict(list)
    nodes = ['tgt']
    node_label = ['tgt']
    node_opacity = [1.0]
    edges = []
    edge_label = []
    edge_directed = []

    una_stats = n_una / n
    if n_una > 0:
        nodes.append('una')
        node_label.append('una')
        node_opacity.append(una_stats)
        r_lab = f'una-lab'
        edges.append(('una', 'tgt'))
        edge_label.append(f'{{una: {una_stats * 100:.1f}\%}}')

    rel_node_opacity = []
    for r in rels:
        r_n = sum(n_rels[r])
        r_p = r_n / n
        rel_stats.append((r_p, r))
        nodes.append(r)
        node_label.append(f'{{{r}: {r_p:.1f}}}')
        rel_node_opacity.append(r_p)
        if r == 'parent':
            edges.append((r, 'tgt'))
        elif r == 'siblings':
            edges.append(('parent', r))
        elif r == 'grandparents':
            edges.append((r, 'parent'))
        elif r == 'aunts':
            edges.append(('grandparents', r))
        elif r == 'child':
            edges.append(('tgt', r))
        elif r == 'coparents':
            edges.append((r, 'child'))
        else:
            raise ValueError(r)
        for l, v in n_rel_labels[r].items():
            if l != 'una' and sum(v) > 0:
                l_p = (sum(v) * 100 / r_n) if r_n > 0 else 0.
                label_stats[r].append((l_p, l))
        edge_label.append(f'{{' + '\\\\'.join([f'{l}: {l_p:.1f}\%' for l_p, l in sorted(label_stats[r], reverse=True)[:5]]) + '}')

    print(f'UNA: {una_stats:.1f}%', '|', ', '.join([f'{r}: {x:.1f}' for x, r in rel_stats]))
    for r in rels:
        if r in label_stats:
            print(f'{r}:', ', '.join([f'{l}: {x:.1f}%' for x, l in sorted(label_stats[r], reverse=True)]))

    max_rel_stat = max(rel_stats)[0]

    style = {}
    style['node_label'] = node_label
    style['node_opacity'] = node_opacity + [x / max_rel_stat for x in rel_node_opacity]
    style['layout'] = {'tgt': (0, 0),
                       'parent': (-3, 4),
                       'siblings': (-6, 0),
                       'grandparents': (-6, 8),
                       'aunts': (-9, 4),
                       'child': (-3, -4),
                       'coparents': (-9, 0),
                       'una': (-3, 0)}
    style['node_color'] = {'tgt': 'red'}
    style['canvas'] = (16, 16)
    style['margin'] = 1

    plot((nodes, []), filename='plot.tex', **style)
    ignore_lines = ['\\documentclass{standalone}',
                    '\\usepackage{tikz-network}',
                    '\\begin{document}',
                    '\\end{tikzpicture}',
                    '\\end{document}']
    with open('plot.tex') as f:
        for line in f:
            line = line.strip()
            if line not in ignore_lines:
                print(line.strip())
    for (a, b), l in zip(edges, edge_label):
        print(f'\\path[->] ({a}) edge node[align=right,shape=rectangle,fill=white] {l} ({b});')
    print('\\end{tikzpicture}')


def eval_sub(logits, labels, tokenizer, top_n=1):
    log_probs = torch.log_softmax(logits, dim=-1).detach()
    sorted_ps, sorted_idxs = torch.sort(log_probs, descending=True)
    sorted_ps = sorted_ps.view(-1, tokenizer.vocab_size)
    sorted_idxs = sorted_idxs.view(-1, tokenizer.vocab_size)

    nll = torch.nn.functional.nll_loss(log_probs, labels, reduction='sum')

    position_correct = torch.max(sorted_idxs == labels.unsqueeze(1).expand(-1, sorted_idxs.size(1)),
                                 dim=-1).indices.float()
    sum_position_correct = torch.sum(position_correct).item()
    sum_reciprocal_rank = torch.sum(1 / (position_correct + 1)).item()
    del position_correct

    sum_entropy = torch.distributions.Categorical(logits=logits).entropy().sum().item()

    ps = []
    rs = []
    pdms = []
    vocab = set()
    for n in range(1, top_n+1):
        p = 100 * n / tokenizer.vocab_size
        ps.append(p)
        recalled = 0
        m = 0
        pdm = sorted_ps[:, :n].exp().sum(dim=-1)
        pdms.append(pdm.sum().item())
        del pdm
        for i, b in enumerate(sorted_idxs[:, :n]):
            idx = labels[i]
            if idx != -100:
                if idx in b:
                    recalled += 1
                idx = idx.item()
                if idx not in vocab:
                    vocab.add(idx)
                m += 1
        rs.append(recalled / m)

    del sorted_ps, sorted_idxs
    torch.cuda.empty_cache()

    return {'count': m, 'conf': pdms[0], 'acc': recalled, 'ppl': nll,
            'avg_position_correct': sum_position_correct, 'mrr': sum_reciprocal_rank,
            'entropy': sum_entropy, 'vocab': vocab}


def eval_snap(predicted, gold, sizes, types):
    sub_pred = torch.split(predicted, sizes, dim=-1)
    sub_gold = torch.split(gold, sizes, dim=-1)
    result = defaultdict(list)
    with torch.no_grad():
        for i, (pre, gol, siz, typ) in enumerate(zip(sub_pred, sub_gold, sizes, types)):
            result[f'{typ}_nonzerodim'].append(((pre.count_nonzero() - gol.count_nonzero()) / siz).item())
            result[f'{typ}_norm'].append((torch.linalg.vector_norm(pre) - torch.linalg.vector_norm(gol)).item())
            result[f'{typ}_cos'].append((cos_sim(pre.reshape(1, -1), torch.ones(1, pre.numel(), device=pre.device)) - \
                                        cos_sim(gol.reshape(1, -1), torch.ones(1, gol.numel(), device=gol.device))).item())
    return result


def eval_all(model, tokenizer, data, domains, device='cpu'):
    model.eval()

    graph_evals = defaultdict(lambda: defaultdict(list))
    lm_evals1 = defaultdict(lambda: defaultdict(list))
    lm_evals2 = defaultdict(lambda: defaultdict(list))
    lm_evals3 = defaultdict(lambda: defaultdict(list))

    for i, (id_batch, x_batch, l_batch, token_batch, first_token_batch) in enumerate(data):
        batch_domains = [domains[_id] for _id in id_batch]
        domain_counts = Counter(batch_domains)
        if len(domain_counts) > 1:
            top_domain = domain_counts.most_common(1)[0][0]
            print(f'WARNING: batch contains multiple different domains: {domain_counts} -> choosing most common one: {top_domain}', file=sys.stderr)
            batch_domain = top_domain
        else:
            batch_domain = batch_domains[0]

        model_outputs = model(l_batch, gm_inputs=x_batch, plm_gm_tokens=token_batch)

        gm_labels = torch.gather(l_batch, 1, token_batch)
        if gm_labels.size(1) != l_batch.size(1) - 1:
            print(f'WARNING: token index mismatch: {l_batch.size(1) - 1}, {gm_labels.size(1)}, {l_batch}, {gm_labels}, {tokenizer.batch_decode(l_batch)}, {tokenizer.batch_decode(gm_labels)}', file=sys.stderr)
            continue

        gm_labels3 = torch.gather(l_batch, 1, first_token_batch)

        gm_labels[gm_labels == tokenizer.eos_token_id] = -100
        gm_labels = gm_labels.view(-1)
        _gm_mask = gm_labels != -100
        gm_labels = gm_labels[_gm_mask]


        graph_outputs = model_outputs.graph_result

        gm_logits = graph_outputs.logits

        _graph_evals = eval_sub(gm_logits, gm_labels, tokenizer)

        del gm_logits


        gm_labels3[gm_labels3 == tokenizer.eos_token_id] = -100
        gm_labels3 = gm_labels3.view(-1)
        _gm_mask3 = gm_labels3 != -100
        torch.cuda.empty_cache()

        for key, value in _graph_evals.items():
            graph_evals['all'][key].append(value)
            graph_evals[batch_domain][key].append(value)

        del x_batch
        torch.cuda.empty_cache()

        lm_labels = l_batch[:, 1:].reshape(-1)

        lm_logits1 = model_outputs.lm_result.logits[:, :-1].reshape(-1, tokenizer.vocab_size)
        lm_logits1 = lm_logits1[lm_labels != -100]

        lm_labels = lm_labels[lm_labels != -100]

        lm_logits2 = torch.gather(torch.cat(
            [torch.zeros(token_batch.size(0), 1, tokenizer.vocab_size, device=device), model_outputs.lm_result.logits],
            dim=1), 1, token_batch.unsqueeze(-1).expand(-1, -1, tokenizer.vocab_size))
        del token_batch
        torch.cuda.empty_cache()
        lm_logits2 = lm_logits2.view(-1, tokenizer.vocab_size)[_gm_mask]

        lm_logits3 = torch.gather(torch.cat(
            [torch.zeros(first_token_batch.size(0), 1, tokenizer.vocab_size, device=device), model_outputs.lm_result.logits],
            dim=1), 1, first_token_batch.unsqueeze(-1).expand(-1, -1, tokenizer.vocab_size))
        del first_token_batch
        torch.cuda.empty_cache()
        lm_logits3 = lm_logits3.view(-1, tokenizer.vocab_size)[_gm_mask3]

        del l_batch

        _lm_evals1 = eval_sub(lm_logits1, lm_labels, tokenizer)
        _lm_evals2 = eval_sub(lm_logits2, gm_labels, tokenizer)
        _lm_evals3 = eval_sub(lm_logits3, gm_labels3, tokenizer)

        del lm_logits1
        torch.cuda.empty_cache()

        del lm_logits2
        torch.cuda.empty_cache()

        del lm_logits3
        del gm_labels, lm_labels, gm_labels3
        torch.cuda.empty_cache()

        for key, value in _lm_evals1.items():
            lm_evals1['all'][key].append(value)
            lm_evals1[batch_domain][key].append(value)

        for key, value in _lm_evals2.items():
            lm_evals2['all'][key].append(value)
            lm_evals2[batch_domain][key].append(value)

        for key, value in _lm_evals3.items():
            lm_evals3['all'][key].append(value)
            lm_evals3[batch_domain][key].append(value)

        print(i, id_batch, file=sys.stderr)

    _result = {'lm1': lm_evals1, 'lm2': lm_evals2, 'lm3': lm_evals3,
               'graph2': graph_evals}
    result = defaultdict(lambda: defaultdict(dict))
    for model, val in _result.items():
        for dmn, val2 in val.items():
            counts = val2['count']
            toks = sum(counts)
            sents = len(counts)
            result[dmn][model]['count'] = toks
            result[dmn][model]['toks/sent'] = toks / sents
            result[dmn][model]['sents'] = sents
            for metric, val3 in val2.items():
                if metric == 'ppl':
                    result[dmn][model][metric] = torch.exp(sum(val3) / toks).item()
                elif metric != 'count':
                    result[dmn][model][metric] = sum(val3) / toks
    return result


def eval_combined(model, tokenizer, data, domains, interesting_n=100, ud_upos=None, device='cpu'):
    model.eval()

    gold_combined_evals = defaultdict(lambda: defaultdict(list))
    auto_combined_evals = defaultdict(lambda: defaultdict(list))
    lm_evals = defaultdict(lambda: defaultdict(list))
    graph_evals = defaultdict(lambda: defaultdict(list))

    gold_combined_pos_evals = defaultdict(lambda: defaultdict(list))
    auto_combined_pos_evals = defaultdict(lambda: defaultdict(list))
    lm_pos_evals = defaultdict(lambda: defaultdict(list))
    graph_pos_evals = defaultdict(lambda: defaultdict(list))

    max_neg_diff = 0.
    min_pos_diff = 0.
    neg_diffs = []
    pos_diffs = []

    for i, (id_batch, x_batch, l_batch, token_batch, first_token_batch) in enumerate(data):
        batch_domains = [domains[_id] for _id in id_batch]
        domain_counts = Counter(batch_domains)
        if len(domain_counts) > 1:
            top_domain = domain_counts.most_common(1)[0][0]
            print(f'WARNING: batch contains multiple different domains: {domain_counts} -> choosing most common one: {top_domain}', file=sys.stderr)
            batch_domain = top_domain
        else:
            batch_domain = batch_domains[0]
        sent_ids = torch.arange(len(id_batch), device=device).unsqueeze(1).expand(-1, l_batch.size(-1) - 1)
        tok_ids = torch.arange(l_batch.size(-1) - 1, device=device).unsqueeze(0).expand(len(id_batch), -1)

        gm_labels = torch.gather(l_batch, 1, token_batch)
        if gm_labels.size(1) != l_batch.size(1) - 1:
            print(f'WARNING: token index mismatch: {l_batch.size(1) - 1}, {gm_labels.size(1)}, {l_batch}, {gm_labels}, {tokenizer.batch_decode(l_batch)}, {tokenizer.batch_decode(gm_labels)}', file=sys.stderr)
            continue
        gm_sent_ids = torch.gather(sent_ids, 1, token_batch - 1)
        gm_tok_ids = torch.gather(tok_ids, 1, token_batch - 1)

        torch.cuda.empty_cache()

        gm_labels[gm_labels == tokenizer.eos_token_id] = -100
        gm_labels = gm_labels.view(-1)
        _gm_mask = gm_labels != -100
        gm_labels = gm_labels[_gm_mask]

        gm_sent_ids = gm_sent_ids.view(-1)[_gm_mask]
        gm_tok_ids = gm_tok_ids.view(-1)[_gm_mask]

        gold_combined_outputs = model(l_batch, gm_inputs=x_batch, plm_gm_tokens=token_batch, softmax=False, c=1.,
                                   loss_fxn=lambda x, y: (torch.nn.functional.nll_loss(x, y), 0, torch.zeros(1), torch.zeros(1)))
        auto_combined_outputs = model(l_batch, gm_inputs=x_batch, plm_gm_tokens=token_batch, softmax=False, c=0.,
                                   loss_fxn=lambda x, y: (torch.nn.functional.nll_loss(x, y), 0, torch.zeros(1), torch.zeros(1)))
        lm_outputs = gold_combined_outputs.lm_result
        _lm_outputs = model(l_batch, gm_inputs=x_batch, plm_gm_tokens=token_batch, softmax=False, use_graph=False,
                                   loss_fxn=lambda x, y: (torch.nn.functional.nll_loss(x, y), 0, torch.zeros(1), torch.zeros(1)))
        graph_outputs = gold_combined_outputs.graph_result

        torch.cuda.empty_cache()

        gold_combined_logits2 = gold_combined_outputs.logits.view(-1, tokenizer.vocab_size)
        auto_combined_logits2 = auto_combined_outputs.logits.view(-1, tokenizer.vocab_size)
        lm_logits2 = lm_outputs.logits[:, :-1].reshape(-1, tokenizer.vocab_size)[_gm_mask]
        _lm_logits2 = _lm_outputs.logits.view(-1, tokenizer.vocab_size)
        graph_logits2 = graph_outputs.logits.view(-1, tokenizer.vocab_size)

        assert lm_logits2.size() == _lm_logits2.size(), (lm_logits2.size(), _lm_logits2.size())
        del _lm_logits2
        torch.cuda.empty_cache()

        del l_batch

        _gold_combined_evals2 = eval_sub(gold_combined_logits2, gm_labels, tokenizer)
        _gold_combined_evals2['lm loss'] = float(gold_combined_outputs.lm_result.loss) * gm_labels.size(0)
        _gold_combined_evals2['graph loss'] = float(gold_combined_outputs.graph_result.loss) * gm_labels.size(0)
        for name, loss in zip(AUX_LOSSES, gold_combined_outputs.aux_losses):
            _gold_combined_evals2[name] = float(loss) * gm_labels.size(0)
        _auto_combined_evals2 = eval_sub(auto_combined_logits2, gm_labels, tokenizer)
        _auto_combined_evals2['lm loss'] = float(auto_combined_outputs.lm_result.loss) * gm_labels.size(0)
        _auto_combined_evals2['graph loss'] = float(auto_combined_outputs.graph_result.loss) * gm_labels.size(0)
        for name, loss in zip(AUX_LOSSES, auto_combined_outputs.aux_losses):
            _auto_combined_evals2[name] = float(loss) * gm_labels.size(0)
        if isinstance(auto_combined_outputs, mtl_result) and auto_combined_outputs.snapshot_eval is not None:
            for name, v in auto_combined_outputs.snapshot_eval.items():
                _auto_combined_evals2[name] = sum(v)
        _lm_evals2 = eval_sub(lm_logits2, gm_labels, tokenizer)
        _graph_evals2 = eval_sub(graph_logits2, gm_labels, tokenizer)

        for key, value in _gold_combined_evals2.items():
            gold_combined_evals['all'][key].append(value)
            gold_combined_evals[batch_domain][key].append(value)

        for key, value in _auto_combined_evals2.items():
            auto_combined_evals['all'][key].append(_auto_combined_evals2[key])
            auto_combined_evals[batch_domain][key].append(_auto_combined_evals2[key])

        for key, value in _lm_evals2.items():
            lm_evals['all'][key].append(_lm_evals2[key])
            lm_evals[batch_domain][key].append(_lm_evals2[key])

        for key, value in _graph_evals2.items():
            graph_evals['all'][key].append(_graph_evals2[key])
            graph_evals[batch_domain][key].append(_graph_evals2[key])

        # TODO: add una-length breakdown
        # TODO: plot sentence-wise NLLs/PPLs across two models

        if ud_upos is not None:
            gold_combined_pos_logits2 = defaultdict(list)
            auto_combined_pos_logits2 = defaultdict(list)
            lm_pos_logits2 = defaultdict(list)
            graph_pos_logits2 = defaultdict(list)
            pos_labels = defaultdict(list)
            for m, (j, k, l) in enumerate(zip(gm_sent_ids, gm_tok_ids, gm_labels)):
                sent_id = id_batch[j]
                pos = ud_upos[sent_id][k+1]
                pos_class = UPOS_CLASSES[pos]
                gold_combined_pos_logits2[pos_class].append(gold_combined_logits2[[m]])
                auto_combined_pos_logits2[pos_class].append(auto_combined_logits2[[m]])
                lm_pos_logits2[pos_class].append(lm_logits2[[m]])
                graph_pos_logits2[pos_class].append(graph_logits2[[m]])
                pos_labels[pos_class].append(l.unsqueeze(0))

            for pos_class, labels in pos_labels.items():
                labels = torch.cat(labels, dim=0)
                _gold_combined_pos_evals = eval_sub(torch.cat(gold_combined_pos_logits2[pos_class], dim=0),
                                                    labels, tokenizer)
                _auto_combined_pos_evals = eval_sub(torch.cat(auto_combined_pos_logits2[pos_class], dim=0),
                                                    labels, tokenizer)
                _lm_pos_evals = eval_sub(torch.cat(lm_pos_logits2[pos_class], dim=0),
                                         labels, tokenizer)
                _graph_pos_evals = eval_sub(torch.cat(graph_pos_logits2[pos_class], dim=0),
                                            labels, tokenizer)
                for key, value in _gold_combined_pos_evals.items():
                    gold_combined_pos_evals[pos_class][key].append(value)
                for key, value in _auto_combined_pos_evals.items():
                    auto_combined_pos_evals[pos_class][key].append(_auto_combined_pos_evals[key])
                for key, value in _lm_pos_evals.items():
                    lm_pos_evals[pos_class][key].append(_lm_pos_evals[key])
                for key, value in _graph_pos_evals.items():
                    graph_pos_evals[pos_class][key].append(_graph_pos_evals[key])

        _interesting_tokens, interesting_nll_diffs = get_interesting_tokens(lm_logits2, gold_combined_logits2,
                                                                            gm_labels,
                                                                            n=interesting_n)
        _interesting_sentences = torch.gather(gm_sent_ids, 0, _interesting_tokens)
        interesting_sentences = [id_batch[x] for x in _interesting_sentences]
        interesting_tokens = torch.gather(gm_tok_ids, 0, _interesting_tokens)
        for d, s, t in zip(interesting_nll_diffs, interesting_sentences, interesting_tokens):
            d = d.item()
            t = t.item()
            if len(neg_diffs) < interesting_n or d < max_neg_diff:
                neg_diffs.append((d, s, t))
                max_neg_diff = max(d, max_neg_diff)
            if len(pos_diffs) < interesting_n or d > min_pos_diff:
                pos_diffs.append((d, s, t))
                min_pos_diff = min(d, min_pos_diff)

        del gm_sent_ids, gm_tok_ids, interesting_tokens, _interesting_tokens, interesting_nll_diffs, _interesting_sentences
        torch.cuda.empty_cache()

        del gold_combined_logits2, auto_combined_logits2, graph_logits2, lm_logits2  #, lm_loss2
        torch.cuda.empty_cache()

        del gm_labels
        torch.cuda.empty_cache()

        print(i, id_batch, file=sys.stderr)

    _result = {'gold2': gold_combined_evals, 'auto2': auto_combined_evals, 'lm2': lm_evals, 'graph2': graph_evals}
    if ud_upos is not None:
        pos_results = {'gold2': gold_combined_pos_evals,
                       'auto2': auto_combined_pos_evals,
                       'lm2': lm_pos_evals,
                       'graph2': graph_pos_evals}
    result = defaultdict(lambda: defaultdict(dict))
    for model, val in _result.items():
        all_items = sorted(val.items())
        if ud_upos is not None:
            all_items += sorted(pos_results[model].items())
        for dmn, val2 in all_items:
            counts = val2['count']
            toks = sum(counts)
            sents = len(counts)
            result[dmn][model]['count'] = toks
            result[dmn][model]['toks/sent'] = toks / sents
            result[dmn][model]['sents'] = sents
            for metric, val3 in val2.items():
                if metric == 'ppl':
                    result[dmn][model][metric] = torch.exp(sum(val3) / toks).item()
                elif metric == 'vocab':
                    result[dmn][model][metric] = len(set.union(*val3))
                elif metric != 'count':
                    result[dmn][model][metric] = sum(val3) / toks
    result['all']['diff_tokens'] = neg_diffs, pos_diffs

    return result, _result
