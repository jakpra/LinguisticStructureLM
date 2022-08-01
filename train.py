import sys
import json
import itertools
import torch
import transformers

from argparse import ArgumentParser

import graphmlp
import lm
import _dgl
from graphmlp import auto_data_loop
from read_mrp import read_mrp_file, DUMMY_LABEL, INV_LABEL, UNA_LABEL
from util import Dir, get_capacities, str2bool

argp = ArgumentParser()
argp.add_argument('formalism', type=str, help='Linguistic formalism; must match a training and validation .mrp file in DATA.')
argp.add_argument('epochs', type=int, help='How many epochs to train for.')
argp.add_argument('train_mode', type=int, help='0: graph, 1: lm, 2: both, 3: combined, '
                                               '4: graph (combined sanity), 5: lm (combined sanity)')
argp.add_argument('from_scratch', type=str2bool, help='Whether or not to train the Transformer language model (GPT-2) from scratch.')
argp.add_argument('per_l', type=str2bool, help='Whether or not to permute edge labels in the input.')
argp.add_argument('per_a', type=str2bool, help='Whether or not to permute token-to-node anchoring in the input.')
argp.add_argument('keep_una', type=str2bool, help='Whether or not to retain unanalyzable (multi-word) anchors when permuting token anchoring (does nothing if PER_A=0).')
argp.add_argument('lm_weight', type=float, help='Weight of LM-finetuning-only loss in addition to ensemble-LM loss.')
argp.add_argument('aux_weight', type=float, help='Weight of MTL auxiliary loss in addition to ensemble-LM loss. (Not implemented)')
argp.add_argument('nslm_no_tokens', type=str2bool, help='Whether or not to remove tokens from linguistic graph inputs.')
argp.add_argument('seed', type=int, help='Seed for random model and data shuffling initialization.')

argp.add_argument('--data', type=str, default='mrp/', help='Main data directory.')
argp.add_argument('--baseline-enc', type=str, choices=['gcn', 'rgcn', 'gat'], default=None, help='Which variant of graph neural net baseline to use, if any.')
argp.add_argument('--upos-file', type=str, default=None, help='Performs comparison-by-combination with UPOS if .mrp file is provided.')

args = argp.parse_args()

upos = None
upos_types = None
if args.upos_file:
    with open(args.upos_file) as f:
        upos = json.load(f)
    upos_types = {t: i for i, t in enumerate(sorted(map(str, set(itertools.chain(*map(set, upos.values()))))))}

lbda = 0

torch.autograd.set_detect_anomaly(True)
transformers.set_seed(args.seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
if args.from_scratch:
    gpt2 = transformers.GPT2LMHeadModel(transformers.GPT2Config.from_pretrained('gpt2'))
else:
    gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
embedding = gpt2.get_input_embeddings()
embedding_dim = embedding.embedding_dim


data_dir = f'{args.data}/{args.formalism}.training.mrp'


_d, edge_labels = read_mrp_file(data_dir, permute_labels=args.per_l, permute_anchors=args.per_a, keep_una=args.keep_una,
                                seed=args.seed)
SHARED_IDS = []
d = [x for x in _d if x['id'] not in SHARED_IDS]

edge_labels[DUMMY_LABEL] = len(edge_labels)
edge_labels[INV_LABEL] = len(edge_labels)
edge_labels[UNA_LABEL] = len(edge_labels)
n_edge_labels = len(edge_labels)
print('edge labels', n_edge_labels, edge_labels)

sibling_dir = Dir.l2r

inp_emb = torch.nn.Embedding.from_pretrained(embedding.weight.detach().cpu())
emb_param = torch.nn.Parameter(inp_emb.weight)

if args.nslm_no_tokens:
    inp_emb = None

max_parents = 1
index_immediates = True
max_children = 1
max_coparents = 0
max_siblings = 1
max_aunts = 1
max_grandparents = 0
max_una = 0
parent_feat_dim = (max_siblings + 1 + max_aunts + 1 + max_grandparents + 1 + 1) * \
                  (max_parents + 1) * len(edge_labels) + \
                  ((max_siblings + 1 + max_aunts + 1 + max_grandparents + 1 + 1) * \
                   (max_parents + 1) + (max_una + 1)) * \
                  embedding_dim * int(inp_emb is not None)
child_feat_dim = (max_coparents + 1 + 1) * (max_children + 1) * len(edge_labels) + \
                 (max_coparents + 1 + 1) * (max_children + 1) * \
                 embedding_dim * int(inp_emb is not None)
feat_dim = parent_feat_dim + child_feat_dim
print('parent_feat_dim', parent_feat_dim)
print('child_feat_dim', child_feat_dim)
print('feat_dim', feat_dim)

capacity_sizes, capacity_types, capacity_rels = get_capacities(len(edge_labels), embedding_dim,
                                                               max_parents=max_parents,
                                                               max_siblings=max_siblings,
                                                               max_grandparents=max_grandparents,
                                                               max_aunts=max_aunts,
                                                               max_children=max_children,
                                                               max_coparents=max_coparents,
                                                               max_una=max_una,
                                                               index_immediates=index_immediates,
                                                               index_tokens=inp_emb is not None,
                                                               index_pos=len(args.upos_types) if upos is not None else 0)
feat_dim = sum(capacity_sizes)
print('feat_dim', feat_dim)


kwargs = dict(embedding_dim=embedding_dim, edge_labels=edge_labels,
              rels=['parent_', 'sibling_s', 'grandparent_s', 'aunt_s', 'child_', 'coparent_s'],
              max_parents=max_parents, index_immediates=index_immediates,
              max_children=max_children, max_coparents=max_coparents,
              max_siblings=max_siblings, max_aunts=max_aunts,
              max_grandparents=max_grandparents)
if args.baseline_enc == 'gcn':
    encoder = _dgl.GNN(_dgl.GraphConv, **kwargs)
    feat_dim = encoder.out_dim
elif args.baseline_enc == 'rgcn':
    encoder = _dgl.GNN(_dgl.RelGraphConv, **kwargs)
    feat_dim = encoder.out_dim
elif args.baseline_enc == 'gat':
    encoder = _dgl.GNN(_dgl.GATConv, **kwargs)
    feat_dim = encoder.out_dim
else:
    encoder = graphmlp.SparseSliceEncoder(**kwargs)  # TODO: add an out_dim to SparseSliceEncoder to unify this

mlp = graphmlp.EmbeddingGraphMLP(feat_dim, 1024, [768], tokenizer, encoder, emb_param, dropout=0.2)

if device == 'cuda':
    mlp.cuda()

batch_size = 8

train_dev_split = int(len(d) * 0.9)

dev_d = d[train_dev_split:]
d = d[:train_dev_split]

data = auto_data_loop(graphmlp.raw_data_loop, d, edge_labels, tokenizer, encoder,
                      upos=upos, upos_types=upos_types,
                      encode_incremental=0,  # (set to 0 to speed up training, so that GNN baselines are only run once per sentence. in case of memory issues, set to >0 (the smaller, the slower and less memory used)
                      device=device,
                      embedding=inp_emb,
                      batch_size=batch_size, write_cache=False,
                      max_una=max_una,
                      sibling_dir=sibling_dir)

dev_data = auto_data_loop(graphmlp.raw_data_loop, dev_d, edge_labels, tokenizer, encoder,
                          upos=upos, upos_types=upos_types,
                          encode_incremental=0,  # (0) to speed up training, GNN baselines are only run once per sentence
                          device=device,
                          embedding=inp_emb,
                          batch_size=1, write_cache=False,
                          return_first_idxs=True,
                          max_una=max_una,
                          sibling_dir=sibling_dir)

if args.train_mode in (3, 4, 5):
    reg_lm = lm.CombinedLM(gpt2, mlp, use_graph=args.train_mode in (3, 4), use_lm=args.train_mode in (3, 5))
elif args.train_mode in (0, 1, 2):
    reg_lm = lm.RegularizedLM(gpt2, mlp)
else:
    raise NotImplementedError

if device == 'cuda':
    reg_lm.cuda()

reg_lm.eval()


if args.train_mode == 0:
    for param in reg_lm.pretrained_lm.parameters():
        param.requires_grad = False
elif args.train_mode == 1:
    for param in reg_lm.graph_model.parameters():
        param.requires_grad = False


checkpoint_name = 'checkpoint.pt'
if args.train_mode in (0, 1, 2):
    checkpoint_name = f'{args.formalism}_model-{args.epochs}-{int(args.from_scratch)}{int(args.per_l)}{int(args.per_a)}{int(args.keep_una)}-{int(args.nslm_no_tokens)}-{args.seed}{args.baseline_enc or ""}.pt'
if args.train_mode in (3, 4, 5):
    checkpoint_name = f'{args.formalism}_combined_model-{args.epochs}{({4: "gm", 5: "lm"}[args.train_mode]) if args.train_mode in (4, 5) else ""}-{int(args.from_scratch)}{int(args.per_l)}{int(args.per_a)}{int(args.keep_una)}-{args.lm_weight}_{args.aux_weight}-{int(args.nslm_no_tokens)}-{args.seed}{args.baseline_enc or ""}.pt'
if args.train_mode in (6, 7):
    checkpoint_name = f'{args.formalism}_multitask_model-{args.epochs}{"lm" if args.train_mode == 7 else ""}-{int(args.from_scratch)}{int(args.per_l)}{int(args.per_a)}{int(args.keep_una)}-{args.lm_weight}_{args.aux_weight}-{int(args.nslm_no_tokens)}-{args.seed}.pt'

loaded = None
try:
    loaded = torch.load(checkpoint_name, map_location='cpu')
except:
    print('model not found.', file=sys.stderr)
if loaded is not None:
    try:
        reg_lm.load_state_dict(loaded)
    except Exception as e:
        print(f'couldn\'t load model: {e}', file=sys.stderr)


lm.train(reg_lm, data, dev_data=dev_data, epochs=args.epochs, n_data=((len(data) // batch_size) + 1), randomize=True,
         seed=args.seed,
         lbda=lbda,
         train_mode=args.train_mode, checkpoint_name=checkpoint_name,
         plm_lr=1e-6, gm_lr=1e-4, lr=1e-4,
         lm_weight=args.lm_weight, gm_weight=0., aux_weight=args.aux_weight,
         loss_fxn=lambda x, y: (torch.nn.functional.nll_loss(x, y), 0, torch.zeros(1), torch.zeros(1)))


reg_lm.eval()

