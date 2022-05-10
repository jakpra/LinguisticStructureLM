import os
import tqdm
import random

import torch

from util import spikes_result, graph_result, \
    get_node_offset, get_offset_diff, \
    aggregate_relatives, get_node_text, Dir


def extract_graph_fragment_for_indexing(node_id, id2node, edge_labels,
                                        node2text=None, tokenizer=None, embedding=None,
                                        sibling_dir=Dir.l2r):
    '''
    Maps the dictionary-based graph representation read from the MRP format to a sparse index vector.

    :param node_id:
    :param id2node:
    :param edge_labels:
    :param max_parents, max_siblings, max_grandparents: How many relatives to track explicitly. n = 0 means that all
    instances will be aggregated; n >= 1 means that encodings for the closest n instances will be concatenated
    and remaining instances m > n will be aggregated.
    :return:
    '''
    assert (node2text is None) == (tokenizer is None) == (embedding is None)

    embedding_dim = embedding.embedding_dim if embedding is not None else 0

    node = id2node[node_id]
    offset = get_node_offset(node_id, id2node)
    parent_contexts = []
    for p_offset, parent_id, label in sorted((get_node_offset(i, id2node), i, l) for i, l in node.get('parents', [])):
        parent = id2node[parent_id]
        parent_idx = edge_labels.get(label, 0)  # TODO: 0 shouldn't be the default; probably dummy_label_idx is better
        parent_token_emb = None
        parent_char_offset = []
        _, p_offs_sign, _ = get_offset_diff(offset, parent_id, id2node)

        if p_offs_sign != 0 and p_offs_sign not in sibling_dir:  # needed for UD and such, where parents might lie
            continue  # entirely in the future (as opposed to containing
            # the target)

        if p_offs_sign in sibling_dir:
            parent_char_offset.append(p_offset[0])
            if node2text is not None:
                parent_word = get_node_text(parent_id, node2text, id2node)
                parent_token_emb = embedding(
                    tokenizer(f" {parent_word} ", return_tensors='pt').input_ids[
                        0]).mean(0).view(1, -1)
            else:
                parent_word = None
        else:
            parent_word = None

        siblings = sorted(
            (get_offset_diff(offset, i, id2node), get_node_offset(i, id2node), i, l) for l, i in parent['children'] if
            i != node_id)
        sibling_idxs = []
        sibling_token_embs = []
        sibling_char_offsets = []
        sibling_words = []
        for (_, offs_sign, _), sib_offs, i, l in siblings:
            if offs_sign in sibling_dir:
                sibling_idxs.append(edge_labels.get(l, 0))
                sibling_char_offsets.append(sib_offs[0])
                if node2text is not None:
                    sibling_word = get_node_text(i, node2text, id2node)
                    sibling_words.append(sibling_word)
                    sibling_token_embs.append(embedding(
                        tokenizer(f" {sibling_word} ", return_tensors='pt').input_ids[
                            0]).mean(0).view(1, -1))
                else:
                    sibling_words.append(None)
            else:
                sibling_words.append(None)
        # TODO: replace tokenizer input with all_target_idxs.char_to_token(get_node_offset(i, id2node) + special_len)
        # TODO: if no siblings, use directly preceding tokens instead?
        c = {'parent': torch.LongTensor([parent_idx]),
             'parent_offset': torch.LongTensor(parent_char_offset),
             'parent_id': (p_offs_sign, p_offset, parent_id, label),
             'parent_word': parent_word,
             'siblings': torch.LongTensor(sibling_idxs),
             'sibling_offsets': torch.LongTensor(sibling_char_offsets),
             'sibling_ids': siblings,
             'sibling_words': sibling_words}
        if node2text is not None:
            c['parent_token'] = parent_token_emb if parent_token_emb is not None else torch.zeros(1, embedding_dim)
            c['sibling_tokens'] = torch.cat(sibling_token_embs, dim=0) if sibling_token_embs else torch.zeros(1,
                                                                                                              embedding_dim)
        if parent.get('parents'):
            grandparents = sorted(
                (get_offset_diff(offset, i, id2node), get_node_offset(i, id2node), i, l) for i, l in parent['parents'])
            gp_idxs = []
            gp_token_embs = []
            gp_char_offsets = []
            grandparent_words = []

            for (_, offs_sign, _), gp_offs, i, l in grandparents:
                gp_idxs.append(edge_labels.get(l, 0))
                if offs_sign in sibling_dir:
                    gp_char_offsets.append(gp_offs[0])
                    if node2text is not None:
                        grandparent_word = get_node_text(i, node2text, id2node)
                        grandparent_words.append(grandparent_word)
                        gp_token_embs.append(embedding(
                            tokenizer(f" {grandparent_word} ", return_tensors='pt').input_ids[
                                0]).mean(0).view(1, -1))
                    else:
                        grandparent_words.append(None)
                else:
                    grandparent_words.append(None)
            c['grandparents'] = torch.LongTensor(gp_idxs)
            c['grandparent_offsets'] = torch.LongTensor(gp_char_offsets)
            c['grandparent_ids'] = grandparents
            c['grandparent_words'] = grandparent_words
            if node2text is not None:
                c['grandparent_tokens'] = torch.cat(gp_token_embs, dim=0) if gp_token_embs else torch.zeros(1,
                                                                                                            embedding_dim)

            aunts = sorted((get_offset_diff(offset, i, id2node), get_node_offset(i, id2node), i, l) for _, _, gp_i, _ in
                           grandparents for l, i in
                           id2node[gp_i]['children'] if i not in (node_id, parent_id))
            aunt_idxs = []
            aunt_token_embs = []
            aunt_char_offsets = []
            aunt_words = []
            for (_, offs_sign, _), aunt_offs, i, l in aunts:
                if offs_sign in sibling_dir:
                    aunt_idxs.append(edge_labels.get(l, 0))
                    aunt_char_offsets.append(aunt_offs[0])
                    if node2text is not None:
                        aunt_word = get_node_text(i, node2text, id2node)
                        aunt_words.append(aunt_word)
                        aunt_token_embs.append(embedding(
                            tokenizer(f" {aunt_word} ", return_tensors='pt').input_ids[
                                0]).mean(0).view(1, -1))
                    else:
                        aunt_words.append(None)
                else:
                    aunt_words.append(None)
            c['aunts'] = torch.LongTensor(aunt_idxs)
            c['aunt_offsets'] = torch.LongTensor(aunt_char_offsets)
            c['aunt_ids'] = aunts
            c['aunt_words'] = aunt_words
            if node2text is not None:
                c['aunt_tokens'] = torch.cat(aunt_token_embs, dim=0) if aunt_token_embs else torch.zeros(1,
                                                                                                         embedding_dim)
        parent_contexts.append(c)

    child_contexts = []
    for c_offset, child_id, label in sorted((get_node_offset(i, id2node), i, l) for l, i in node.get('children', [])):
        child = id2node[child_id]
        child_idx = edge_labels.get(label, 0)
        child_token_emb = None
        child_char_offset = []

        _, c_offs_sign, _ = get_offset_diff(offset, child_id, id2node)
        if c_offs_sign != 0 and c_offs_sign not in sibling_dir:  # needed for UD and such, where children might lie
            continue  # entirely in the future (as opposed to containing
            # the target)

        if c_offs_sign in sibling_dir:
            child_char_offset.append(c_offset[0])
            if node2text is not None:
                child_word = get_node_text(child_id, node2text, id2node)
                child_token_emb = embedding(
                    tokenizer(f" {child_word} ", return_tensors='pt').input_ids[
                        0]).mean(0).view(1, -1)
            else:
                child_word = None
        else:
            child_word = None

        coparents = sorted(
            (get_offset_diff(offset, i, id2node), get_node_offset(i, id2node), i, l) for i, l in child['parents'] if
            i != node_id)
        cp_idxs = []
        cp_token_embs = []
        cp_char_offsets = []
        cp_words = []

        for (_, offs_sign, _), cp_offs, i, l in coparents:
            cp_idxs.append(edge_labels.get(l, 0))
            if offs_sign in sibling_dir:
                cp_char_offsets.append(cp_offs[0])
                if node2text is not None:
                    cp_word = get_node_text(i, node2text, id2node)
                    cp_words.append(cp_word)
                    cp_token_embs.append(embedding(
                        tokenizer(f" {cp_word} ", return_tensors='pt').input_ids[
                            0]).mean(0).view(1, -1))
                else:
                    cp_words.append(None)
            else:
                cp_words.append(None)

        c = {'child': torch.LongTensor([child_idx]),
             'child_offset': torch.LongTensor(child_char_offset),
             'child_id': (c_offs_sign, c_offset, child_id, label),
             'child_word': child_word,
             'coparents': torch.LongTensor(cp_idxs),
             'coparent_offsets': torch.LongTensor(cp_char_offsets),
             'coparent_ids': coparents,
             'coparent_words': cp_words}
        if node2text is not None:
            c['child_token'] = child_token_emb if child_token_emb is not None else torch.zeros(1, embedding_dim)
            c['coparent_tokens'] = torch.cat(cp_token_embs, dim=0) if cp_token_embs else torch.zeros(1, embedding_dim)

        child_contexts.append(c)

    return parent_contexts, child_contexts


class SparseSliceEncoder:
    def __init__(self, embedding_dim, edge_labels, *args,
                 max_parents=2, index_immediates=True,
                 max_children=2, max_coparents=0,
                 max_siblings=5, max_aunts=0,
                 max_grandparents=0, **kwargs):
        del args, kwargs

        self.embedding_dim = embedding_dim
        self.n_dim = len(edge_labels)
        self.max_parents = max_parents
        self.index_immediates = index_immediates
        self.max_children = max_children
        self.max_coparents = max_coparents
        self.max_siblings = max_siblings
        self.max_aunts = max_aunts
        self.max_grandparents = max_grandparents

    def gather_contexts(self, parent_contexts, child_contexts, **kwargs):
        del kwargs

        if parent_contexts:
            indexed = []
            for context in parent_contexts:
                this_indexed = []
                if self.index_immediates:
                    p = torch.nn.functional.one_hot(context['parent'], num_classes=self.n_dim).float()
                else:
                    p = torch.ones(1, self.n_dim) / self.n_dim  # to distinguish from having no parents
                this_indexed.append(p)
                s = torch.nn.functional.one_hot(context['siblings'], num_classes=self.n_dim).float()
                s = aggregate_relatives(s, self.max_siblings)
                this_indexed.append(s)
                if 'grandparents' in context:
                    g = torch.nn.functional.one_hot(context['grandparents'], num_classes=self.n_dim).float()
                    g = aggregate_relatives(g, self.max_grandparents)
                    a = torch.nn.functional.one_hot(context['aunts'], num_classes=self.n_dim).float()
                    a = aggregate_relatives(a, self.max_aunts)
                else:
                    g = torch.zeros(1, (self.max_grandparents + 1) * self.n_dim)
                    a = torch.zeros(1, (self.max_aunts + 1) * self.n_dim)
                this_indexed.append(g)
                this_indexed.append(a)
                if self.embedding_dim is not None:
                    pt = context['parent_token']
                    this_indexed.append(pt)
                    st = aggregate_relatives(context['sibling_tokens'], self.max_siblings)
                    this_indexed.append(st)
                    if 'grandparents' in context:
                        gpt = aggregate_relatives(context['grandparent_tokens'], self.max_grandparents)
                        at = aggregate_relatives(context['aunt_tokens'], self.max_aunts)
                    else:
                        gpt = torch.zeros(1, (self.max_grandparents + 1) * self.embedding_dim)
                        at = torch.zeros(1, (self.max_aunts + 1) * self.embedding_dim)
                    this_indexed.append(gpt)
                    this_indexed.append(at)
                indexed.append(torch.cat(this_indexed, dim=1).float())
        else:
            feat_dim = (self.max_siblings + 1 + self.max_aunts + 1 + self.max_grandparents + 1 + \
                        1) * self.n_dim + \
                       (self.max_siblings + 1 + self.max_aunts + 1 + self.max_grandparents + 1 + \
                        1) * \
                       self.embedding_dim * int(self.embedding_dim is not None)
            indexed = [torch.zeros(1, feat_dim)]
        parent_result = aggregate_relatives(torch.cat(indexed, dim=0), self.max_parents)

        if child_contexts:
            indexed = []
            for context in child_contexts:
                this_indexed = []
                if self.index_immediates:
                    c = torch.nn.functional.one_hot(context['child'], num_classes=self.n_dim).float()
                else:
                    c = torch.ones(1, self.n_dim) / self.n_dim  # to distinguish from having no children
                this_indexed.append(c)
                cp = torch.nn.functional.one_hot(context['coparents'], num_classes=self.n_dim).float()
                cp = aggregate_relatives(cp, self.max_coparents)
                this_indexed.append(cp)
                if self.embedding_dim is not None:
                    ct = context['child_token']
                    this_indexed.append(ct)
                    cpt = aggregate_relatives(context['coparent_tokens'], self.max_coparents)
                    this_indexed.append(cpt)
                indexed.append(torch.cat(this_indexed, dim=1).float())
        else:
            feat_dim = (self.max_coparents + 1 + 1) * self.n_dim + \
                       (self.max_coparents + 1 + 1) * \
                       self.embedding_dim * int(self.embedding_dim is not None)
            indexed = [torch.zeros(1, feat_dim)]
        child_result = aggregate_relatives(torch.cat(indexed, dim=0), self.max_children)

        return torch.cat([parent_result, child_result], dim=1)

    @staticmethod
    def pre_encode(gathered, incremental=1):
        if incremental == 0 or not gathered:
            return None, gathered
        else:
            return torch.cat(gathered, dim=0), []

    def collect_una(self, tgt_idx, una_embs, max_una=2):
        una_t = aggregate_relatives(torch.cat(una_embs, dim=0) if una_embs else torch.empty(0, self.embedding_dim),
                                    max_una)
        return tgt_idx, una_t

    @staticmethod
    def encode_with_una(encoded, pre_encoded, una_idx, upos_idx=None, incremental=1):
        if incremental == 0:
            return encoded, pre_encoded, una_idx, upos_idx
        else:
            if upos_idx:
                assert len(upos_idx) == len(una_idx), (len(upos_idx), len(una_idx))
            unseen_preencoded = set() if pre_encoded is None else set(range(pre_encoded.size(0)))
            for i, (tgt_idx, una_emb) in enumerate(una_idx):
                components = [pre_encoded[[tgt_idx]], una_emb]
                if upos_idx:
                    components.append(upos_idx[i])
                encoded.append(torch.cat(components, dim=-1))
                unseen_preencoded.discard(tgt_idx)
            assert not unseen_preencoded, (unseen_preencoded, [tgt for tgt, _ in una_idx])

            del pre_encoded, una_idx
            torch.cuda.empty_cache()

            return encoded, None, [], []


def index_graph_fragment(node_id, id2node, edge_labels,
                         encoder,
                         # out of SparseSliceEncoder, _dgl.GraphConv, _dgl.RelGraphConv, _dgl.GATConv  (or make this a superclass of those)
                         node2text=None, tokenizer=None, embedding=None,
                         sibling_dir=Dir.l2r,
                         return_human_readable=False):
    parent_contexts, child_contexts = extract_graph_fragment_for_indexing(node_id, id2node, edge_labels,
                                                                          node2text=node2text,
                                                                          tokenizer=tokenizer,
                                                                          embedding=embedding,
                                                                          sibling_dir=sibling_dir)

    result = encoder.gather_contexts(parent_contexts, child_contexts,
                                     node_id=node_id, id2node=id2node)

    if return_human_readable:
        return result, parent_contexts, child_contexts
    else:
        return result


def get_spikes(t):
    with torch.no_grad():
        s, si = t.sort(descending=True, dim=-1)
        m, mi = s.diff(dim=-1).min(dim=-1)
        mask = t >= s.gather(1, mi.unsqueeze(-1))
        counts = mi + 1

        return spikes_result(counts=counts.float(),
                             indices=mask,
                             diff=m)


def positive_reinforcement_nllloss(decay=.01, min_cluster_ratio=5, ignore_index=-100):
    def l(y_hat, y):
        mask = y != ignore_index
        y = y.unsqueeze(-1)
        y = y[mask]
        y_hat = y_hat[mask]

        exp_y_hat = torch.exp(y_hat)
        cluster_sizes, cluster_indices, cluster_deltas = get_spikes(y_hat)

        cluster_ratios = torch.exp(-cluster_deltas)
        cluster_indices[cluster_ratios < min_cluster_ratio] = False
        cluster_indices.scatter_(1, y, True)
        updated_cluster_sizes = torch.count_nonzero(cluster_indices, dim=-1).float()
        thr = torch.log(1 / updated_cluster_sizes)
        positive_reinforcement = torch.mean(torch.clamp_min(thr - y_hat.gather(1, y).squeeze(-1), min=0), dim=0)

        decay_losses = -torch.log(1 - exp_y_hat) * (2 - exp_y_hat.detach())
        decay_losses[cluster_indices] *= decay
        decay_losses.scatter_(1, y, 0.)
        decay_loss = torch.mean(torch.sum(decay_losses, dim=1), dim=0)
        return positive_reinforcement, decay_loss, updated_cluster_sizes, cluster_ratios

    return l


class GraphMLP(torch.nn.Module):
    '''Produces a distribution over an external tokenizer's vocabulary based on an encoded graph fragment.'''

    def __init__(self, in_dim, h_dim1, h_dims, tokenizer, encoder, dropout=0.2):
        super(GraphMLP, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.hidden1 = torch.nn.Linear(in_dim, h_dim1)
        self.hiddens = torch.nn.ModuleList()
        last_hidden = h_dim1
        for h_dim in h_dims:
            self.hiddens.append(torch.nn.Linear(last_hidden, h_dim))
            last_hidden = h_dim
        self.last_hidden = last_hidden
        self.out = torch.nn.Linear(self.last_hidden, self.tokenizer.vocab_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def _map_to_vocab(self, hidden):
        return self.out(hidden)

    def forward(self, indexed_context, labels=None, softmax=True,
                loss_fxn=positive_reinforcement_nllloss()):
        h = self.activation(self.dropout(self.hidden1(indexed_context)))
        for hidden in self.hiddens:
            h = self.activation(self.dropout(hidden(h))).clone()
        out = self._map_to_vocab(h)
        out_softmax = torch.log_softmax(out, dim=-1)
        if labels is not None:
            loss1, loss2, cluster_sizes, cluster_ratios = loss_fxn(out_softmax, labels)
            loss = loss1 + loss2
            cluster_size = cluster_sizes.sum().item()
            cluster_ratio = cluster_ratios.sum().item()
        else:
            loss = cluster_size = cluster_ratio = None
        return graph_result(logits=out_softmax if softmax else out,
                            loss=loss,
                            last_hidden_state=h,
                            cluster_size=cluster_size,
                            cluster_ratio=cluster_ratio)


class EmbeddingGraphMLP(GraphMLP):
    '''Produces a distribution over an external tokenizer's vocabulary based on an encoded graph fragment
    and using pretrained embeddings.'''

    def __init__(self, in_dim, h_dim1, h_dims, tokenizer, encoder, embedding, dropout=0.05):
        super(EmbeddingGraphMLP, self).__init__(in_dim, h_dim1, h_dims, tokenizer, encoder, dropout=dropout)
        self.embedding_weights = torch.nn.Linear(embedding.size(1), self.tokenizer.vocab_size)
        self.embedding_weights.weight = embedding
        self.out = torch.nn.Linear(self.last_hidden, embedding.size(1))

    def _map_to_vocab(self, hidden):
        emb = self.out(hidden)
        return self.embedding_weights(emb)


def raw_data_loop(data, edge_labels, tokenizer, encoder, batch_size,
                  upos=None, upos_types=None,
                  encode_incremental=True,
                  cache_dir='cache', write_cache=False,
                  embedding=None,
                  return_first_idxs=False,
                  max_una=2,
                  sibling_dir=Dir.l2r):
    if not os.path.exists(f'{cache_dir}/x'):
        os.makedirs(f'{cache_dir}/x')
        os.makedirs(f'{cache_dir}/y')
    elif data is None and not os.listdir(f'{cache_dir}/x') and not write_cache:
        raise ValueError('Cache directory is empty.')

    id_batch = []
    x_batch = []
    l_batch = []
    token_batch = []
    first_token_batch = []
    token_mismatches = []
    special_len = len('<|endoftext|>')

    if upos is not None:
        n_upos_types = len(upos_types)

    for datum in data:
        text_id = datum['id']
        all_target_idxs = tokenizer(f"<|endoftext|>{datum['text']}", return_tensors='pt')
        n = len(all_target_idxs.input_ids[0])
        token_idxs = []
        first_token_idxs = []
        encodeds = []
        gathereds = []
        una_idxs = []
        id2node = datum['id2node']
        node2text = datum['node2text']
        char_node = sorted(datum['chars2node'].items())

        upos_idxs = None
        if upos is not None:
            text_upos = upos[text_id]
            upos_idxs = []

        for (char_idx, node_ids), (next_char_idx, _) in zip(char_node, char_node[1:] + [(None, None)]):

            tok_idx = all_target_idxs.char_to_token(char_idx + special_len)
            next_tok_idx = all_target_idxs.char_to_token(
                next_char_idx + special_len) if next_char_idx is not None else n

            if next_tok_idx == tok_idx:
                continue  # ensures that each GPT token is covered by not more than one graph node (I think)
                # does the order of precedence make sense though?
                # depends on chars2node sorting, which in turn depends on node_ids

            node_ids = (((len(id2node[x].get('parents', [])),
                          len(id2node[x].get('children', []))), x) for x in node_ids)
            # TODO: by most-parents-then-LEAST-children to get lowest most informative node? (currently most-parents-most-children)
            _node_ids = sorted([(x, y) for x, y in node_ids], reverse=True)  # if x > (0, 0)
            if not _node_ids:
                print(datum)
                print(char_idx, tok_idx)
                print(next_char_idx, next_tok_idx)
                print(list(node_ids))
                raise ValueError
            node_id = _node_ids[0][1]

            gathered, *human_readable_contexts = index_graph_fragment(node_id, id2node, edge_labels, encoder,
                                                                      node2text=node2text if embedding is not None else None,
                                                                      tokenizer=tokenizer if embedding is not None else None,
                                                                      embedding=embedding if embedding is not None else None,
                                                                      sibling_dir=sibling_dir,
                                                                      return_human_readable=True)

            first_token_idxs.append(torch.LongTensor([tok_idx]))

            gathereds.append(gathered)
            pre_encoded, gathereds = encoder.pre_encode(gathereds, incremental=encode_incremental)

            una_tokens = []
            human_readable_una = []
            at_least_one_tok = False
            for token_idx in range(tok_idx, next_tok_idx):
                at_least_one_tok = True

                aligned_target_idx = all_target_idxs.input_ids[0, token_idx]
                human_readable_tgt = tokenizer.decode(aligned_target_idx)
                token_idxs.append(torch.LongTensor([token_idx]))
                n_gathered = len(gathereds) if pre_encoded is None else pre_encoded.size(0)
                una_idx = encoder.collect_una(n_gathered - 1,
                                              una_tokens,
                                              max_una=max_una)
                una_idxs.append(una_idx)

                if upos is not None:
                    tok_upos = text_upos[token_idx]
                    upos_idxs.append(torch.nn.functional.one_hot(torch.LongTensor([upos_types[tok_upos]]),
                                                                 num_classes=n_upos_types).float())

                una_tokens.append(embedding(aligned_target_idx).view(1, -1))
                human_readable_una.append((token_idx, aligned_target_idx, human_readable_tgt))
            assert at_least_one_tok, (tok_idx, next_tok_idx)
            assert n_gathered > 0 or not una_idxs, ([tgt for tgt, _ in una_idx], n_gathered)
            assert len(una_idxs) >= n_gathered, ([tgt for tgt, _ in una_idxs], n_gathered)
            encodeds, pre_encoded, una_idxs, upos_idxs = encoder.encode_with_una(encodeds, pre_encoded, una_idxs,
                                                                                 upos_idx=upos_idxs,
                                                                                 incremental=encode_incremental)

        pre_encoded, gathereds = encoder.pre_encode(gathereds)
        encodeds = encoder.encode_with_una(encodeds, pre_encoded, una_idxs, upos_idx=upos_idxs)[0]
        source_idxs = torch.cat(encodeds, dim=0)
        token_idxs = torch.cat(token_idxs, dim=0)
        first_token_idxs = torch.cat(first_token_idxs, dim=0)
        if write_cache:
            with open(f'{cache_dir}/x/{text_id}.pt', 'wb') as f:
                torch.save(source_idxs, f)
            with open(f'{cache_dir}/l/{text_id}.pt', 'wb') as f:
                torch.save(all_target_idxs.input_ids[0], f)
            with open(f'{cache_dir}/t/{text_id}.pt', 'wb') as f:
                torch.save(token_idxs, f)
        id_batch.append(text_id)
        x_batch.append(source_idxs)
        l_batch.append(all_target_idxs.input_ids[0])
        token_batch.append(token_idxs)
        first_token_batch.append(first_token_idxs)
        if len(l_batch) >= batch_size > 0:
            x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
            l_batch = torch.nn.utils.rnn.pad_sequence(l_batch, batch_first=True, padding_value=-100)
            token_batch = torch.nn.utils.rnn.pad_sequence(token_batch, batch_first=True)
            first_token_batch = torch.nn.utils.rnn.pad_sequence(first_token_batch, batch_first=True)
            if return_first_idxs:
                yield id_batch, x_batch, l_batch, token_batch, first_token_batch
            else:
                yield id_batch, x_batch, l_batch, token_batch
            id_batch = []
            x_batch = []
            char_batch = []
            l_batch = []
            token_batch = []
            first_token_batch = []
    if l_batch:
        x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        l_batch = torch.nn.utils.rnn.pad_sequence(l_batch, batch_first=True, padding_value=-100)
        token_batch = torch.nn.utils.rnn.pad_sequence(token_batch, batch_first=True)
        first_token_batch = torch.nn.utils.rnn.pad_sequence(first_token_batch, batch_first=True)

        if return_first_idxs:
            yield id_batch, x_batch, l_batch, token_batch, first_token_batch
        else:
            yield id_batch, x_batch, l_batch, token_batch


class auto_data_loop:
    '''wrapper for raw_data_loop that moves datapoints to and from GPU as needed'''
    def __init__(self, looper, *args, device='cpu', **kwargs):
        self.looper = looper
        self.item = None
        self.data = args[0]
        self.args = args[1:]
        self.kwargs = kwargs
        self.device = device
        self.interactive = False
        self.loop = self.looper(self.data, *self.args, **self.kwargs)

    def set_interactive(self, true_or_false):
        self.interactive = true_or_false

    def __next__(self):
        try:
            item = next(self.loop)
        except StopIteration:
            self.loop = self.looper(self.data, *self.args, **self.kwargs)
            raise
        else:
            if self.device == 'cuda':
                if self.item is not None:
                    for x in self.item:
                        del x
                    torch.cuda.empty_cache()
                item = tuple((x.cuda() if isinstance(x, torch.Tensor) else x) for x in item)
            self.item = item
            return self.item

    def __iter__(self):
        return self

    def __getitem__(self, key):
        if self.interactive:
            data = self.data[key] if isinstance(key, slice) else [self.data[key]]
            return self.looper(data, *self.args, **self.kwargs)
        return self.data[key]

    def __setitem__(self, key, value):
        if self.interactive:
            raise NotImplementedError
        self.data[key] = value

    def __len__(self):
        return len(self.data)


def cache_loop(cache_dir, batch_size, randomize=True):
    '''deprecated, to delete'''
    x_batch = []
    y_batch = []
    c_batch = []
    l_batch = []
    files = os.listdir(f'{cache_dir}/x')
    if randomize:
        random.shuffle(files)
    for f in files:
        x_batch.append(torch.load(f'{cache_dir}/x/{f}'))
        y_batch.append(torch.load(f'{cache_dir}/y/{f}'))
        c_batch.append(torch.load(f'{cache_dir}/c/{f}'))
        l_batch.append(torch.load(f'{cache_dir}/l/{f}'))
        if len(y_batch) >= batch_size > 0:
            yield torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True), \
                  torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True), \
                  torch.nn.utils.rnn.pad_sequence(c_batch, batch_first=True), \
                  torch.nn.utils.rnn.pad_sequence(l_batch, batch_first=True)
            x_batch = []
            y_batch = []
            c_batch = []
            l_batch = []
    if y_batch:
        yield torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True), \
              torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True), \
              torch.nn.utils.rnn.pad_sequence(c_batch, batch_first=True), \
              torch.nn.utils.rnn.pad_sequence(l_batch, batch_first=True)


def train(model, data, n_data=None, randomize=True, seed=42, epochs=50, lr=1e-3,
          loss_fxn=positive_reinforcement_nllloss()):
    optim = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=.05)
    model.train()
    random.seed(seed)
    if n_data is None:
        try:
            n_data = len(data)
        except:
            pass
    with tqdm.tqdm(None, total=epochs, desc=f'Total', unit_scale=True) as total_pbar:
        for i in range(epochs):
            with tqdm.tqdm(None, desc=f'Total - Epoch {i + 1}', total=epochs) as pbar:
                pbar.update(i + 1)
                total_pbar.set_description(f'Total - Epoch {i + 1}')
                loss = 0.
                cluster_size = 0.
                cluster_ratio = 0.
                n = 0
                if randomize:
                    random.shuffle(data)
                with tqdm.tqdm(data, total=n_data, desc=f'Epoch {i + 1}') \
                        as pbar_batch:
                    for _id, x_batch, y_batch, _ in pbar_batch:
                        optim.zero_grad()
                        model_outputs = model(x_batch.view(-1, x_batch.size(-1)), labels=y_batch[:, 1:].view(-1),
                                              loss_fxn=loss_fxn)
                        batch_n = x_batch.size(0)
                        n += 1
                        loss += model_outputs.loss
                        cluster_size += model_outputs.cluster_size
                        cluster_ratio += model_outputs.cluster_ratio
                        batch_loss = model_outputs.loss  # / batch_n
                        batch_loss.backward()
                        optim.step()
                        pbar_batch.set_postfix(batch_loss=batch_loss.item(),
                                               batch_cl_size=model_outputs.cluster_size / batch_n,
                                               batch_cl_ratio=model_outputs.cluster_ratio / batch_n,
                                               text_ids=list(_id))
                        pbar.set_postfix(total_loss=loss.item() / n,
                                         total_cl_size=cluster_size / n,
                                         total_cl_ratio=cluster_ratio / n)
                        if n_data is not None:
                            total_pbar.update(1 / n_data)
