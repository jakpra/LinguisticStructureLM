import sys
from collections import defaultdict
import itertools
import json
import copy

import datetime
from argparse import ArgumentParser

import torch
import transformers

import evaluation
import graphmlp
import lm
import _dgl
from graphmlp import auto_data_loop
from read_mrp import read_mrp_file, DUMMY_LABEL, INV_LABEL, UNA_LABEL
from util import Dir, get_capacities
from evaluation import eval_combined, get_all_structure_stats, approx_rand_significance


# torch.use_deterministic_algorithms(True)
# #torch.set_deterministic_debug_mode(1)
# if torch.backends.cudnn.is_available():
#     print('cuDNN ok')
#     torch.backends.cudnn.determinism = True
# print('Determinism is switched', 'ON' if torch.are_deterministic_algorithms_enabled() else 'OFF')

SYN_SEM = {
    'ud': 'syntax',
    'ptb-phrase': 'syntax',
    'ptb-func': 'syntax',
    'ptb-all': 'syntax',
    'ptb-pos': 'syntax',
    'dm': 'semantics',
    'psd': 'semantics',
    'eds': 'semantics',
    'ptg': 'semantics'
}

DEP_CON = {
    'ud': 'dependency',
    'dm': 'dependency',
    'psd': 'dependency',
    'ptb-phrase': 'constituency',
    'ptb-func': 'constituency',
    'ptb-all': 'constituency',
    'ptb-pos': 'constituency',
    'eds': 'constituency',
    'ptg': 'constituency'
}


argp = ArgumentParser()
argp.add_argument('formalisms', type=str, help='Comma-separated list of linguistic formalisms; each must match a training and validation .mrp file in DATA.')
argp.add_argument('tags', type=str, help='Comma-separated list of config tags, each following the format -EPOCHS-FLAGS[-WEIGHTS[-TOKENFLAG]]-SEED where EPOCHS is the number of epochs, FLAGS is a sequence of 4 bits corresponding to FROM_SCRATCH, PER_L, PER_A, and KEEP_UNA in train.py, WEIGHTS is LMWEIGHT_AUXWEIGHT (train.py), and TOKENFLAG is NSLM_NO_TOKENS (train.py).')
argp.add_argument('models', type=str, help='Comma-separated list of model types (combined, graph, lm)')

argp.add_argument('--eval-upos-file', type=str, default=None, help='Analyzes performance breakdown by UPOS if .mrp file is provided.')
argp.add_argument('--baseline-enc', type=str, choices=['gcn', 'rgcn', 'gat'], default=None, help='Which variant of graph neural net baseline to use, if any.')
argp.add_argument('--train-upos-file', type=str, default=None, help='Performs comparison-by-combination with UPOS if .mrp file is provided.')

args = sys.argv[1:]
clean_args = []
tags = None
for i, arg in enumerate(args):
    if arg.startswith('-'):
        if i == 1:
            tags = arg
            clean_args.append(arg.strip('-'))
        elif i > 2:
            clean_args.append(arg)
        else:
            raise Exception(f'encountered unexpected dash arg at position {i+1}: {arg}')
    else:
        clean_args.append(arg)

args = argp.parse_args(clean_args)
args.tags = tags



formalisms = args.formalisms.split(',')
_tags = []
tags = args.tags.split(',')
for t in tags:
    t = t.split('-')
    if len(t) >= 4:
        epochs, flags, seed = t[1:4]
        weights = '?'
    if len(t) == 5:
        weights, seed = t[3:5]
    elif len(t) == 6:
        weights, nslm_no_tokens, seed = t[3:6]
    else:
        raise ValueError
    if len(flags) == 4:
        from_scratch, per_l, per_a, keep_una = [bool(int(x)) for x in flags]
    elif len(flags) == 1:
        from_scratch, per_l, per_a, keep_una = bool(int(flags)), False, False, True
    else:
        raise ValueError
    _tags.append((weights, from_scratch, per_l, per_a, keep_una, epochs, seed))

eval_upos = None
if args.eval_upos_file:
    with open(args.eval_upos_file) as f:
        eval_upos = json.load(f)

input_upos = False
train_upos = None
upos_types = None
if args.train_upos_file:
    input_upos = True
    with open(args.train_upos_file) as f:
        train_upos = json.load(f)
    upos_types = {t: i for i, t in enumerate(sorted(map(str, set(itertools.chain(*map(set, train_upos.values()))))))}


batch_size = 8
val_batch_size = 1
interesting_n = 100


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


now = lambda: datetime.datetime.now(datetime.timezone.utc)


tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
embedding = gpt2.get_input_embeddings()
embedding_dim = embedding.embedding_dim
emb_param = torch.nn.Parameter(embedding.weight)  # for GraphMLP


print('Date added;Model;Edge labels;Input dim;Syntax/Semantics;Dependency/ConstituencyTrainable params;Training data;'
      'Training domains;Training sents;Training toks;Training toks/sent;Epochs;Batch size;Seed;Loss weights;'
      'From scratch;Random labels;Random anchors;Keep UNA;Eval data;Eval data sub;Eval data filter;Eval sents;'
      'Eval toks;Eval toks/sent;Eval vocab;Eval tok filter;POS class;Accuracy;Perplexity;Avg Pos Correct;'
      'Mean Recip Rank;Confidence;Entropy;LM Loss;Graph Loss;Aux Loss;'
      'hi_res_label_nonzerodim;hi_res_label_norm;hi_res_label_cos;'
      'lo_res_label_nonzerodim;lo_res_label_norm;lo_res_label_cos;'
      'emb_nonzerodim;emb_norm;emb_cos;')

print()

evals_for_stat_testing = {}

for formalism in formalisms:

    print(formalism)

    val_data_dir = f'mrp/{formalism}.validation.mrp'
    data_dir = f'mrp/{formalism}.training.mrp'

    d, edge_labels = read_mrp_file(data_dir)

    edge_labels[DUMMY_LABEL] = len(edge_labels)
    edge_labels[INV_LABEL] = len(edge_labels)
    edge_labels[UNA_LABEL] = len(edge_labels)
    # print(edge_labels)

    train_sents = len(d)
    train_toks = 0

    train_domains = set()
    for x in d:
        train_domains.add(x['source'])
    train_domains = ','.join(sorted(train_domains))

    lm_all_params = sum([p.numel() for p in gpt2.parameters()])
    lm_grad_params = sum([p.numel() for p in gpt2.parameters() if p.requires_grad])
    print('lm all', lm_all_params)
    print('lm grad', lm_grad_params)

    max_parents = 1
    index_immediates = True
    max_children = 1
    max_coparents = 0
    max_siblings = 1
    max_aunts = 1
    max_grandparents = 0
    max_una = 0
    sibling_dir = Dir.l2r

    kwargs = dict(embedding_dim=embedding_dim, edge_labels=edge_labels,
                  rels=['parent_', 'sibling_s', 'grandparent_s', 'aunt_s', 'child_', 'coparent_s'],
                  max_parents=max_parents, index_immediates=index_immediates,
                  max_children=max_children, max_coparents=max_coparents,
                  max_siblings=max_siblings, max_aunts=max_aunts,
                  max_grandparents=max_grandparents)
    if args.baseline_enc == 'gcn':
        encoder = _dgl.GNN(_dgl.GraphConv, **kwargs)
        graph_in_dim = encoder.out_dim
    elif args.baseline_enc == 'rgcn':
        encoder = _dgl.GNN(_dgl.RelGraphConv, **kwargs)
        graph_in_dim = encoder.out_dim
    elif args.baseline_enc == 'gat':
        encoder = _dgl.GNN(_dgl.GATConv, **kwargs)
        graph_in_dim = encoder.out_dim
    else:
        encoder = graphmlp.SparseSliceEncoder(**kwargs)
        graph_in_dim = None

    last_known_working = {}

    all_evals1 = defaultdict(dict)
    all_evals2 = defaultdict(dict)
    all_evals3 = defaultdict(dict)

    for tag, (weights, from_scratch, per_l, per_a, keep_una, epochs, seed) in zip(tags, _tags):

        if nslm_no_tokens is not None:
            nslm_no_tokens = bool(int(nslm_no_tokens))
        elif weights != '?' and '_' not in weights:
            nslm_no_tokens = bool(int(weights))
        else:
            nslm_no_tokens = False

        val_d, _ = read_mrp_file(val_data_dir, permute_labels=per_l, permute_anchors=per_a, keep_una=keep_una)
        # val_d, _ = read_mrp_file(val_data_dir)  # for train shuffle only
        # # TODO: debugging only!! // hand-picked example
        # val_d = [x for x in val_d if x['id'] == '22102004']
        # val_d = [x for x in val_d if x['id'] == '22170053']

        val_domains = {}
        for x in val_d:
            val_domains[x['id']] = x['source']

        capacity_sizes, capacity_types, capacity_rels = get_capacities(len(edge_labels), embedding.embedding_dim,
                                                                       max_parents=max_parents,
                                                                       max_siblings=max_siblings,
                                                                       max_grandparents=max_grandparents,
                                                                       max_aunts=max_aunts,
                                                                       max_children=max_children,
                                                                       max_coparents=max_coparents,
                                                                       max_una=max_una,
                                                                       index_immediates=index_immediates,
                                                                       index_tokens=not nslm_no_tokens,
                                                                       index_pos=len(upos_types) if input_upos else 0)
        feat_dim = sum(capacity_sizes)

        mlp = graphmlp.EmbeddingGraphMLP(graph_in_dim or feat_dim, 1024, [768], tokenizer, encoder, emb_param, dropout=0.2)

        graph_all_params = sum([p.numel() for p in mlp.parameters()])
        graph_grad_params = sum([p.numel() for p in mlp.parameters() if p.requires_grad])

        evals = {}
        other_evals = {}

        if args.models is None or 'graph' in args.models:
            loaded = None
            try:
                model_file = f'{formalism}_graph_model{tag}{args.baseline_enc or ""}.pt'
                loaded = torch.load(model_file, map_location='cpu')
            except:
                model_file = last_known_working.get('graph')
                if model_file:
                    print(f'graph model not found, backing up to last known working: {model_file}', file=sys.stderr)
                    loaded = torch.load(model_file, map_location='cpu')
                else:
                    print('graph model not found.', file=sys.stderr)
            if loaded is not None:
                try:
                    mlp.load_state_dict(loaded)
                except Exception as e:
                    print(f'couldn\'t load graph model: {e}', file=sys.stderr)
                else:
                    last_known_working['graph'] = model_file

            print('graph all', graph_all_params)
            print('graph grad', graph_grad_params)

        if args.models is None or 'lm' in args.models:
            loaded = None
            try:
                model_file = f'{formalism}_lm_model{tag}.pt'
                loaded = torch.load(model_file, map_location='cpu')
            except:
                model_file = last_known_working.get('lm')
                if model_file:
                    print(f'lm model not found, backing up to last known working: {model_file}', file=sys.stderr)
                    loaded = torch.load(model_file, map_location='cpu')
                else:
                    print('lm model not found.', file=sys.stderr)
            if loaded is not None:
                try:
                    gpt2.load_state_dict(loaded)
                except Exception as e:
                    print(f'couldn\'t load lm model: {e}', file=sys.stderr)
                else:
                    last_known_working['lm'] = model_file

        if args.models is None or 'lm' in args.models or 'graph' in args.models:
            reg_lm = lm.RegularizedLM(copy.deepcopy(gpt2), copy.deepcopy(mlp))
            if device == 'cuda':
                reg_lm.cuda()
            reg_lm.eval()
            embedding = reg_lm.pretrained_lm.get_input_embeddings()
            inp_emb = None if nslm_no_tokens else torch.nn.Embedding.from_pretrained(embedding.weight.cpu())

            val_data = auto_data_loop(graphmlp.raw_data_loop, val_d, edge_labels, tokenizer, reg_lm.graph_model.encoder,
                                      batch_size=val_batch_size,
                                      upos=eval_upos if input_upos else None, upos_types=upos_types if input_upos else None,
                                      encode_incremental=0,
                                      device=device,
                                      embedding=copy.deepcopy(inp_emb),
                                      return_first_idxs=True,
                                      write_cache=False,
                                      max_una=max_una,
                                      sibling_dir=sibling_dir)
            evals, _ = eval_combined(reg_lm, tokenizer, val_data, val_domains,
                                     interesting_n=interesting_n, ud_upos=eval_upos, device=device)
            if args.models is not None and ('lm' not in args.models or 'graph' not in args.models):
                dmns = list(evals.keys())
                for dmn in dmns:
                    mdls = list(evals[dmn].keys())
                    for mdl in mdls:
                        if 'lm' not in args.models and mdl.startswith('lm'):
                            evals[dmn].pop(mdl)
                        if 'graph' not in args.models and mdl.startswith('graph'):
                            evals[dmn].pop(mdl)

        if args.models is None or 'combined' in args.models:
            loaded = None
            combined_lm = lm.CombinedLM(copy.deepcopy(gpt2), copy.deepcopy(mlp), use_lm=not epochs.endswith('gm'), use_graph=not epochs.endswith('lm'))
            model_file = f'{formalism}_combined_model{tag}{args.baseline_enc or ""}.pt'
            try:
                loaded = torch.load(model_file, map_location='cpu')
            except:
                model_file = last_known_working.get('combined')
                if model_file:
                    print(f'combined model not found, backing up to last known working: {model_file}', file=sys.stderr)
                    loaded = torch.load(model_file, map_location='cpu')
                else:
                    print('combined model not found.', file=sys.stderr)
            if loaded is not None:
                try:
                    combined_lm.load_state_dict(loaded)
                except Exception as e:
                    print(f'couldn\'t load combined model: {e}', file=sys.stderr)
                else:
                    last_known_working['combined'] = model_file

            if device == 'cuda':
                combined_lm.cuda()
            combined_lm.eval()

            cm_all_params = sum([p.numel() for p in combined_lm.parameters()])
            cm_grad_params = sum([p.numel() for p in combined_lm.parameters() if p.requires_grad])
            print('cm all', cm_all_params)
            print('cm grad', cm_grad_params)

            other_evals['combined'] = {'params': cm_grad_params}
            embedding = combined_lm.pretrained_lm.get_input_embeddings()
            inp_emb = None if nslm_no_tokens else torch.nn.Embedding.from_pretrained(embedding.weight.cpu())
            val_data = auto_data_loop(graphmlp.raw_data_loop, val_d, edge_labels, tokenizer, combined_lm.graph_model.encoder,
                                      batch_size=val_batch_size,
                                      upos=eval_upos if input_upos else None, upos_types=upos_types if input_upos else None,
                                      encode_incremental=0,
                                      device=device,
                                      embedding=copy.deepcopy(inp_emb),
                                      return_first_idxs=True,
                                      write_cache=False,
                                      max_una=max_una, sibling_dir=sibling_dir)
            ce, disaggr_ce = eval_combined(combined_lm, tokenizer, val_data, val_domains,
                                           interesting_n=interesting_n, ud_upos=eval_upos, device=device)
            neg_diff_tokens, pos_diff_tokens = ce['all'].pop('diff_tokens')
            other_evals['combined']['domains'] = ce
            evals_for_stat_testing[model_file] = disaggr_ce

            neg_diff_tokens = sorted(neg_diff_tokens)[:interesting_n]
            pos_diff_tokens = sorted(pos_diff_tokens, reverse=True)[:interesting_n]

            print('Graph Improvement graph stats:')
            neg_diff_stats = get_all_structure_stats(neg_diff_tokens, val_d, edge_labels, tokenizer)

            print('Graph Worse graph stats:')
            pos_diff_stats = get_all_structure_stats(pos_diff_tokens, val_d, edge_labels, tokenizer)

        fmlsm = formalism.split('/')[-1]

        for dmn in evals:
            all_evals1[f'{formalism};{dmn}'][f'gpt2;-;{embedding.embedding_dim};-;-;{lm_grad_params};{formalism};{train_domains};{train_sents};?;?;{epochs};{batch_size};{seed};{weights};{from_scratch};{per_l};{per_a};{keep_una}'] = evals[dmn]['lm1']
            all_evals2[f'{formalism};{dmn}'][f'gpt2;-;{embedding.embedding_dim};-;-;{lm_grad_params};{formalism};{train_domains};{train_sents};?;?;{epochs};{batch_size};{seed};{weights};{from_scratch};{per_l};{per_a};{keep_una}'] = evals[dmn]['lm2']
            all_evals2[f'{formalism};{dmn}'][f'{formalism}-nslm;{len(edge_labels)};{feat_dim};{SYN_SEM.get(fmlsm)};{DEP_CON.get(fmlsm)};{graph_grad_params};{formalism};{train_domains};{train_sents};?;?;{epochs};{batch_size};{seed};{weights};{from_scratch};{per_l};{per_a};{keep_una}'] = evals[dmn]['graph2']
            all_evals3[f'{formalism};{dmn}'][f'gpt2;-;{embedding.embedding_dim};-;-;{lm_grad_params};{formalism};{train_domains};{train_sents};?;?;{epochs};{batch_size};{seed};{weights};{from_scratch};{per_l};{per_a};{keep_una}'] = evals[dmn]['lm3']

        for k, v in other_evals.items():
            for dmn, v2 in v['domains'].items():
                for k2 in v2:
                    if k2.endswith('2'):
                        k3 = k2[:-1]
                        all_evals2[f'{formalism};{dmn}'][f'{formalism}-{k}{f"-{k3}" if k3 else ""};{len(edge_labels)};{feat_dim};{SYN_SEM.get(fmlsm)};{DEP_CON.get(fmlsm)};{v["params"]};{formalism} {train_domains};{train_sents};?;?;{epochs};{batch_size};{seed};{weights};{from_scratch};{per_l};{per_a};{keep_una}'] = v2[k2]

        del mlp
        torch.cuda.empty_cache()

    snapshot_analysis = ['hi_res_label_nonzerodim', 'hi_res_label_norm', 'hi_res_label_cos',
                         'lo_res_label_nonzerodim', 'lo_res_label_norm', 'lo_res_label_cos',
                         'emb_nonzerodim', 'emb_norm', 'emb_cos']

    for dmn, eval in all_evals1.items():
        for name, eval2 in eval.items():
            print(now(), name,
                  dmn, eval2.get('sents'), eval2.get('count'), eval2.get('toks/sent'), eval2.get('vocab'),
                  'all',
                  eval2.get('acc'), eval2.get('ppl'), eval2.get('avg_position_correct'), eval2.get('mrr'),
                  eval2.get('conf'), eval2.get('entropy'),
                  eval2.get('lm loss'), eval2.get('graph loss'), *(eval2.get(x) for x in evaluation.AUX_LOSSES),
                  *(eval2.get(x) for x in snapshot_analysis),
                  sep=';')

    for dmn, eval in all_evals2.items():
        for name, eval2 in eval.items():
            print(now(), name,
                  dmn, eval2.get('sents'), eval2.get('count'), eval2.get('toks/sent'), eval2.get('vocab'),
                  'graph anchors only',
                  eval2.get('acc'), eval2.get('ppl'), eval2.get('avg_position_correct'), eval2.get('mrr'),
                  eval2.get('conf'), eval2.get('entropy'),
                  eval2.get('lm loss'), eval2.get('graph loss'), *(eval2.get(x) for x in evaluation.AUX_LOSSES),
                  *(eval2.get(x) for x in snapshot_analysis),
                  sep=';')

    for dmn, eval in all_evals3.items():
        for name, eval2 in eval.items():
            print(now(), name,
                  dmn, eval2.get('sents'), eval2.get('count'), eval2.get('toks/sent'), eval2.get('vocab'),
                  'first tok per graph anchor only',
                  eval2.get('acc'), eval2.get('ppl'), eval2.get('avg_position_correct'), eval2.get('mrr'),
                  eval2.get('conf'), eval2.get('entropy'),
                  eval2.get('lm loss'), eval2.get('graph loss'), *(eval2.get(x) for x in evaluation.AUX_LOSSES),
                  *(eval2.get(x) for x in snapshot_analysis),
                  sep=';')

    print()

print()

counts = list(evals_for_stat_testing.values())[0]['gold2']['all']['count']

for model1, val1 in evals_for_stat_testing.items():
    for model2, val2 in evals_for_stat_testing.items():
        # TODO: in order to do this for non-all breakdowns, need to rerun eval breakdown within approx_rand loop
        ppl_p = approx_rand_significance(val1['gold2']['all']['ppl'], val2['gold2']['all']['ppl'], counts, aggregate='exp_mean')
        acc_p = approx_rand_significance(val1['gold2']['all']['acc'], val2['gold2']['all']['acc'], counts)
        print(model1, model2, f'ppl p={ppl_p},', f'acc p={acc_p}')
