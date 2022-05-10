import sys
import io
from collections import namedtuple, Counter, defaultdict
import math
import itertools
import contextlib
import torch


spikes_result = namedtuple('spikes_result', ['counts', 'indices', 'diff'])
graph_result = namedtuple('graph_result', ['logits', 'loss', 'last_hidden_state', 'cluster_size', 'cluster_ratio'])
gcn_result = namedtuple('graph_result', ['logits', 'loss', 'last_hidden_state', 'updated_nodes'])
reg_result = namedtuple('reg_result', ['logits', 'aux_loss', 'loss', 'lm_result', 'graph_result', 'aux_losses'])
combined_result = namedtuple('combined_result', ['logits', 'loss', 'aux_loss', 'last_hidden_state', 'lm_result',
                                                 'graph_result', 'aux_losses'])
mtl_result = namedtuple('mtl_result', ['logits', 'loss', 'aux_loss', 'snapshot', 'last_hidden_state', 'lm_result',
                                       'graph_result', 'aux_losses', 'snapshot_eval'])

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception(f'Boolean value expected (yes, no, true, false, y, n, t, f, 1, 0), got {v}.')


class TestOracle(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, t):
        return graph_result(logits=torch.nn.functional.one_hot(t).float(),
                            loss=None, cluster_size=None, cluster_ratio=None)


class TestOracle2(torch.nn.Module):
    def __init__(self, tokenizer, gold):
        super().__init__()
        self.tokenizer = tokenizer
        self.gold = gold

    def forward(self, t):
        result = t.clone()
        result.scatter_(1, self.gold.view(result.size(0), 1), result.size(1))
        return graph_result(logits=torch.softmax(result, dim=1),
                            loss=None, cluster_size=None, cluster_ratio=None)


class AutoEncoderDataSampler:
    def __init__(self, idx_dim, p_multi=0.0, n_sample=10):
        self.idx_dim = idx_dim
        self.p_multi = p_multi
        self.n_sample = n_sample

    def sample(self, n):
        base = torch.nn.functional.one_hot(torch.randint(0, self.idx_dim, (n,)), num_classes=self.idx_dim)
        if self.p_multi > 0.0 and self.n_sample > 1:
            for _ in range(self.n_sample):
                p = torch.rand(n, 1) < (self.p_multi / self.n_sample)
                add = torch.nn.functional.one_hot(torch.randint(0, self.idx_dim, (n,)), num_classes=self.idx_dim)
                base += p.long() * add
        return base.float()

    def sample_batches(self, n_batches, batch_size=1):
        for _ in range(n_batches):
            yield self.sample(batch_size)


class Dir:
    l2r = (-1,)
    r2l = (1,)
    both = (l2r, r2l)


def ispunct(s):
    return all(not c.isalnum() for c in s)


def get_node_offset(node_id, id2node, explored=set()):  # TODO: add a 'lowest' / 'allow_higher' option
    if node_id in explored:  # cycle!!!
        return []
    node = id2node[node_id]
    if 'anchors' in node:
        return sorted(itertools.chain.from_iterable((a['from'], a['to']) for a in node['anchors']))
    new_explored = explored | set([node_id])
    return sorted(itertools.chain.from_iterable(get_node_offset(child_id, id2node, new_explored) for _, child_id in node['children']))


def get_node_text(node_id, node2text, id2node, explored=set()):
    if node_id in explored:  # cycle!!!
        return ''
    node = id2node[node_id]
    if node_id in node2text:
        return node2text[node_id]
    new_explored = explored | set([node_id])
    return ' '.join(map((lambda x: get_node_text(x[1], node2text, id2node, new_explored)),
                        sorted(node['children'],
                               key=(lambda x: get_node_offset(x[1], id2node)))
                       )
                   )


def get_offset_diff(main_offset, node_id, id2node):  # TODO: add a 'lowest' / 'allow_higher' option
    other_offset = get_node_offset(node_id, id2node)
    if not other_offset:
        return math.inf, 1, [math.inf]
    ms, me = main_offset[0], main_offset[-1] - 1
    os, oe = other_offset[0], other_offset[-1] - 1
    offset_diff = (oe - ms) if oe < ms else (os - me) if os > me else 0
    abs_offset_diff = abs(offset_diff)
    return (abs_offset_diff,  # for sorting
            (offset_diff / abs_offset_diff) if abs_offset_diff != 0 else 0,   # sign, for checking direction
            other_offset)


def aggregate_relatives(indexed, max_n=0):
    if max_n == -1:
        return torch.zeros(1, 0)
    n_rel, n_dim = indexed.size()
    max_rel = min(n_rel, max_n)
    aggregated = torch.cat([indexed[:max_rel].view(1, -1),
                            # normalizing
                            torch.sum(indexed[max_rel:], dim=0, keepdim=True) / max(1., (n_rel - max_rel)),
                            torch.zeros(1, max(max_n - n_rel, 0) * n_dim)
                          ], dim=1)
    assert aggregated.size(1) == (max_n + 1) * n_dim, (n_dim, aggregated.size(1), max_n)
    return aggregated


def compute_structure_stats(node_id, id2node, node2text, sibling_dir=Dir.l2r):
    node = id2node[node_id]
    offset = get_node_offset(node_id, id2node)

    rels = ['parent', 'siblings', 'grandparents', 'aunts', 'child', 'coparents']
    local_n_rels = Counter()
    local_n_rel_labels = defaultdict(Counter)

    parent_contexts = []
    for p_offset, parent_id, label in sorted((get_node_offset(i, id2node), i, l) for i, l in node.get('parents', [])):
        parent = id2node[parent_id]
        siblings = sorted(
            (get_offset_diff(offset, i, id2node), get_node_offset(i, id2node), i, l) for l, i in parent['children'] if
            i != node_id)
        sibling_labels = []
        for (_, offs_sign, _), sib_offs, i, l in siblings:
            if offs_sign in sibling_dir:
                sibling_labels.append(l)
        # TODO: if no siblings, use directly preceding tokens instead?
        c = {'parent': [label],
             'siblings': sibling_labels}
        c['parent text'] = get_node_text(parent_id, node2text, id2node)

        if parent.get('parents'):
            grandparents = sorted((get_node_offset(i, id2node), i, l) for i, l in parent['parents'])
            grandparent_labels = [l for _, _, l in grandparents]
            c['grandparents'] = grandparent_labels
            aunts = sorted(
                (get_offset_diff(offset, i, id2node), get_node_offset(i, id2node), i, l) for _, gp_i, _ in grandparents
                for l, i in
                id2node[gp_i]['children'] if i not in (node_id, parent_id))
            aunt_labels = []
            for (_, offs_sign, _), aunt_offs, i, l in aunts:
                if offs_sign in sibling_dir:
                    aunt_labels.append(l)
            c['aunts'] = aunt_labels
        for k, v in c.items():
            if k in rels:
                local_n_rels[k] += len(v)
                local_n_rel_labels[k] += Counter(v)

    child_contexts = []
    for c_offset, child_id, label in sorted((get_node_offset(i, id2node), i, l) for l, i in node.get('children', [])):
        child = id2node[child_id]
        coparents = sorted((get_offset_diff(offset, i, id2node), i, l) for i, l in child['parents'] if i != node_id)
        coparent_labels = []
        for (_, offs_sign, _), i, l in coparents:
            coparent_labels.append(l)
        c = {'child': [label],
             'coparents': coparent_labels}
        c['child text'] = get_node_text(child_id, node2text, id2node)
        for k, v in c.items():
            if k in rels:
                local_n_rels[k] += len(v)
                local_n_rel_labels[k] += Counter(v)

    return local_n_rels, local_n_rel_labels


# for convenience in jupyiter notebooks
@contextlib.contextmanager
def noout():
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    except:
        sys.stdout = save_stdout
        sys.stderr = save_stderr
    sys.stdout = save_stdout
    sys.stderr = save_stderr


def get_capacities(edge_label_dim, embedding_dim, max_parents, max_siblings, max_grandparents, max_aunts, max_children,
                   max_coparents, max_una, index_immediates=True, index_tokens=True, index_pos=0):
    hrl = 'hi_res_label'
    lrl = 'lo_res_label'
    emb = 'emb'
    parent_label_sizes = [edge_label_dim] * (1 + max_siblings + 1 + max_grandparents + 1 + max_aunts + 1)
    parent_label_types = [hrl] * 1 + list(
        itertools.chain(*[([hrl] * n + [lrl]) for n in (max_siblings, max_grandparents, max_aunts)]))
    parent_label_rels = ['p'] * 1 + list(
        itertools.chain(*[([r] * n + [r]) for r, n in zip('bot', (max_siblings, max_grandparents, max_aunts))]))
    parent_token_sizes = [embedding_dim] * (1 + max_siblings + 1 + max_grandparents + 1 + max_aunts + 1)
    parent_token_types = [emb] * (1 + max_siblings + 1 + max_grandparents + 1 + max_aunts + 1)
    parent_token_rels = parent_label_rels[:]
    parent_sizes = (parent_label_sizes + parent_token_sizes * int(index_tokens)) * max_parents \
                   + [edge_label_dim] * len(parent_label_sizes) \
                   + [embedding_dim] * len(parent_token_sizes) * int(index_tokens)
    parent_types = (parent_label_types + parent_token_types * int(index_tokens)) * max_parents \
                   + [lrl] * len(parent_label_types) \
                   + [emb] * len(parent_token_types) * int(index_tokens)
    parent_rels = (parent_label_rels + parent_token_rels * int(index_tokens)) * max_parents \
                   + ['p'] * len(parent_label_rels) \
                   + ['p'] * len(parent_token_rels) * int(index_tokens)

    child_label_sizes = [edge_label_dim] * (1 + max_coparents + 1)
    child_label_types = [hrl] * 1 + [hrl] * max_coparents + [lrl]
    child_label_rels = ['c'] * 1 + ['r'] * max_coparents + ['r']
    child_token_sizes = [embedding_dim] * (1 + max_coparents + 1)
    child_token_types = [emb] * (1 + max_coparents + 1)
    child_token_rels = child_label_rels[:]
    child_sizes = (child_label_sizes + child_token_sizes * int(index_tokens)) * max_children \
                  + [edge_label_dim] * len(child_label_sizes) \
                  + [embedding_dim] * len(child_token_sizes) * int(index_tokens)
    child_types = (child_label_types + child_token_types * int(index_tokens)) * max_children \
                  + [lrl] * len(child_label_types) \
                  + [emb] * len(child_token_types) * int(index_tokens)
    child_rels = (child_label_rels + child_token_rels * int(index_tokens)) * max_children \
                  + ['c'] * len(child_label_types) \
                  + ['c'] * len(child_token_types) * int(index_tokens)

    una_sizes = [embedding_dim] * (max_una + 1) * int(index_tokens)
    una_types = [emb] * (max_una + 1) * int(index_tokens)
    una_rels = ['una'] * (max_una + 1) * int(index_tokens)

    sizes = parent_sizes + child_sizes + una_sizes
    types = parent_types + child_types + una_types
    rels = parent_rels + child_rels + una_rels
    if index_pos:
        sizes.append(index_pos)
        types.append(hrl)
        rels.append('pos')
    if len(sizes) != len(types):
        print(len(sizes), len(types), file=sys.stderr)
        for x in (sizes, types, parent_sizes, parent_types, child_sizes, child_types, una_sizes, una_types):
            print(x, file=sys.stderr)
        assert False

    return sizes, types, rels


def magnitude(m):
    return torch.linalg.norm(m, dim=-1)


def cos_sim(m1, m2, clamp_zero=True, eps=1e-8):
    prod = m1 @ m2.t()
    mag1 = magnitude(m1)
    mag2 = magnitude(m2)
    mag = mag1.unsqueeze(1) @ mag2.unsqueeze(0)
    sim = prod / torch.clamp_min(mag, eps)
    if clamp_zero:
        return torch.clamp_min(sim, 0)
    else:
        return sim


def get_batch_seq_mask(batch_size, seq_len):
    seq_m = 1 - torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
    batch_m = torch.diag(torch.ones(batch_size))
    return seq_m.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, batch_size, -1) * batch_m.unsqueeze(1).unsqueeze(3)
