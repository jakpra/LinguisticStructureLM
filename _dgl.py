import sys
import copy

import torch
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, RelGraphConv, GATConv

from read_mrp import INV_LABEL, UNA_LABEL, DUMMY_LABEL


class GNN(torch.nn.Module):
    def __init__(self, gnn_class, embedding_dim, edge_labels, *args,
                 rels=['parent_', 'sibling_s', 'grandparent_s', 'child_', 'coparent_s'],  # 'aunts_s'
                 h_dim=None, n_layers=2, activation=F.relu, n_attn_heads=1, max_bases=None, **kwargs):
        del args, kwargs
        super().__init__()
        self.gnn_class = gnn_class
        self.h_dim = embedding_dim if h_dim is None else h_dim
        self.embedding_dim = embedding_dim
        self.out_dim = self.h_dim + self.embedding_dim  # graph + una
        self.edge_labels = edge_labels
        self.inv_lab_idx = self.edge_labels[INV_LABEL]
        self.una_lab_idx = self.edge_labels[UNA_LABEL]
        self.dummy_lab_idx = self.edge_labels[DUMMY_LABEL]
        self.n_labels = len(self.edge_labels)
        self.rels = rels
        self.activation = activation
        self.n_layers = 2 * n_layers  # by default we assume that edge labels are represented as dummy nodes

        args = {}

        self.lab_projector = None
        self.add_dummy_node = True
        if gnn_class == GraphConv:
            self.lab_projector = torch.nn.Linear(self.n_labels, self.h_dim)
        if gnn_class == RelGraphConv:
            args['num_rels'] = 2 * self.n_labels  # * 2 to account for inverses
            args['regularizer'] = None  # 'basis' # some bug with basis matrix reg leads to mem leak???
            args['num_bases'] = max(self.n_labels // 5, 1)  # equiv to 10% of the original label set
            if max_bases is not None:
                args['num_bases'] = min(args['num_bases'], max_bases)
            self.n_layers //= 2  # when using RGCN, edge labels need not be represented as dummy nodes
            self.add_dummy_node = False
        if gnn_class == GATConv:
            assert n_attn_heads is not None
            args['num_heads'] = n_attn_heads
            self.lab_projector = torch.nn.Linear(self.n_labels, self.h_dim)

        self.layers = torch.nn.ModuleList(
            [self.gnn_class(self.h_dim, self.h_dim, **args) for _ in range(self.n_layers)])
        self.tok_projector = torch.nn.Linear(self.embedding_dim, self.h_dim) \
            if self.h_dim != self.embedding_dim else lambda x: x

        self.I = 1

    def forward(self, graph_idx):
        node_embeddings, edge_matrix, edge_labels_or_dummynode_ids = graph_idx

        num_nodes = node_embeddings.size(0)
        assert edge_matrix.min() >= 0
        assert edge_matrix.max() < num_nodes, (edge_matrix.max(), num_nodes, edge_matrix)
        assert edge_matrix.max() == num_nodes - 1, (edge_matrix.max(), num_nodes, edge_matrix)

        graph = dgl.graph((edge_matrix[0], edge_matrix[1]), num_nodes=num_nodes)
        h = node_embeddings

        for layer in self.layers:
            if self.gnn_class == RelGraphConv:
                h = layer(graph, h, edge_labels_or_dummynode_ids)
            else:
                h = layer(graph, h)
            if self.gnn_class == GATConv:
                h = h.mean(dim=-2)

        del node_embeddings, edge_matrix, edge_labels_or_dummynode_ids, graph
        torch.cuda.empty_cache()

        return h

    def add_node(self, graph_idx, embedding=None, add_self_loop=False):
        device = next(self.parameters()).device

        node_embeddings, edge_matrix, edge_labels_or_dummynode_ids = graph_idx
        new_node_id = len(node_embeddings)
        if embedding is None:
            embedding = torch.zeros(1, self.h_dim, device=device)
        assert embedding.numel() == self.h_dim, embedding.numel()
        node_embeddings.append(embedding.view(1, self.h_dim))
        if add_self_loop:
            if not self.add_dummy_node:
                edge_labels_or_dummynode_ids.append(self.dummy_lab_idx)
            edge_matrix.append([new_node_id, new_node_id])
        return new_node_id, graph_idx

    def add_edge(self, src, tgt, lab, graph_idx, bidirectional=True):
        device = next(self.parameters()).device

        node_embeddings, edge_matrix, edge_labels_or_dummynode_ids = graph_idx
        if self.add_dummy_node:
            lab_enc = F.one_hot(torch.LongTensor([lab]), num_classes=self.n_labels).to(dtype=torch.float, device=device)
            embedding = self.lab_projector(lab_enc)
            new_node_id, graph_idx = self.add_node(graph_idx, embedding=embedding)
            edge_labels_or_dummynode_ids.append(new_node_id)
            edge_matrix.append([src, new_node_id])
            edge_matrix.append([new_node_id, tgt])
            if bidirectional:
                inv_embedding = self.lab_projector(lab_enc +
                                                   F.one_hot(torch.LongTensor([self.inv_lab_idx]),
                                                             num_classes=self.n_labels).to(dtype=torch.float,
                                                                                           device=device))
                new_node_id, graph_idx = self.add_node(graph_idx, embedding=inv_embedding)
                edge_labels_or_dummynode_ids.append(new_node_id)
                edge_matrix.append([tgt, new_node_id])
                edge_matrix.append([new_node_id, src])
        else:
            edge_labels_or_dummynode_ids.append(lab)
            edge_matrix.append([src, tgt])
            if bidirectional:
                edge_labels_or_dummynode_ids.append(lab + self.n_labels)
                edge_matrix.append([tgt, src])
        return graph_idx

    def gather_contexts(self, parent_contexts, child_contexts,
                        node_id, id2node,
                        add_orig_edges=True, **kwargs):
        del kwargs
        device = next(self.parameters()).device

        graph_idx = [], [], []  # node_embeddings, edge_matrix, edge_labels_or_dummynode_ids
        tgt_node_idx, graph_idx = self.add_node(graph_idx,
                                                add_self_loop=True)  # target needs a self loop in case slice is empty
        seen_ids = {node_id: tgt_node_idx}
        edges_to_consider = set()
        for c in parent_contexts + child_contexts:
            for rel in self.rels:
                try:
                    labs = c[rel.replace('_', '')]
                except KeyError:
                    continue

                _ids = c[rel.replace('_', '_id')]
                tok_embs = c[rel.replace('_', '_token')]
                if not rel.endswith('_s'):
                    labs = [labs]
                    _ids = [_ids]
                    tok_embs = [tok_embs]
                for i, (lab, _id) in enumerate(zip(labs, _ids)):
                    _, _, _id, _ = _id

                    if _id not in seen_ids:
                        if i < len(tok_embs):
                            tok_emb = self.tok_projector(tok_embs[i].to(device))
                        else:
                            tok_emb = None
                        node_idx, graph_idx = self.add_node(graph_idx, embedding=tok_emb)
                        seen_ids[_id] = node_idx

                    if add_orig_edges:
                        # edges among relatives
                        node = id2node[_id]
                        for parent, parent_label in node['parents']:
                            edge = (parent, self.edge_labels.get(parent_label, 0), _id)
                            if edge not in edges_to_consider:
                                edges_to_consider.add(edge)
                        for child_label, child in node['children']:
                            edge = (_id, self.edge_labels.get(child_label, 0), child)
                            if edge not in edges_to_consider:
                                edges_to_consider.add(edge)

        # slice should be a connected subgraph, so if there are nodes beyond the target in the slice,
        # they should be attached to something
        node_embeddings, edge_matrix, edge_labels_or_dummynode_ids = graph_idx
        unattached_node_idxs = set(seen_ids.values())
        unattached_node_idxs.discard(tgt_node_idx)
        for src_id, lab_idx, tgt_id in sorted(edges_to_consider, key=lambda x: seen_ids.get(x[0], -1)):
            if src_id in seen_ids and tgt_id in seen_ids:
                src_idx = seen_ids[src_id]
                tgt_idx = seen_ids[tgt_id]
                graph_idx = self.add_edge(src_idx, tgt_idx, lab, graph_idx)
                unattached_node_idxs.discard(src_idx)
                unattached_node_idxs.discard(tgt_idx)
        idx2id = {v: k for k, v in seen_ids.items()}
        if unattached_node_idxs:
            print('target', tgt_node_idx, id2node[node_id], file=sys.stderr)
            print('node_id:idx', seen_ids, file=sys.stderr)
            for c in parent_contexts + child_contexts:
                print('context', c, file=sys.stderr)
            print('edges', edge_matrix, file=sys.stderr)
            for i in unattached_node_idxs:
                print('unattached', i, id2node[idx2id[i]], file=sys.stderr)
            raise Exception

        return (*graph_idx, tgt_node_idx)

    def collect_una(self, tgt_idx, una_embs, **kwargs):
        del kwargs
        device = next(self.parameters()).device

        una_emb = torch.cat(una_embs, dim=0).mean(0).view(1, -1).to(device) \
            if una_embs else torch.zeros(1, self.embedding_dim, device=device)
        # TODO: consider using util.aggregate(..., max_una) here to make it more parallel with sparse enc,
        #       but with the current hyperparameters (max_una=0) that's equivalent anyways

        return tgt_idx, una_emb

    def pre_encode(self, gathered, incremental=1):  # incremental=0 -> all at once; inc>=1 -> once every inc
        device = next(self.parameters()).device

        if incremental == 0 or self.I < incremental or not gathered:
            return None, gathered
        else:
            node_embeddings, edge_matrix, edge_labels_or_dummynode_ids, tgt_idx = [], [], [], []
            node_idx_offset = 0
            for ne, em, eldi, ti in gathered:
                ne = torch.cat(ne, dim=0)
                em = torch.LongTensor(em).to(device) + node_idx_offset
                eldi = torch.LongTensor(eldi).to(device)
                if self.add_dummy_node:
                    eldi += node_idx_offset
                ti += node_idx_offset

                node_embeddings.append(ne)
                edge_matrix.append(em)
                edge_labels_or_dummynode_ids.append(eldi)
                tgt_idx.append(ti)

                node_idx_offset += len(ne)

            node_embeddings = torch.cat(node_embeddings, dim=0)
            edge_matrix = torch.cat(edge_matrix, dim=0).t()
            edge_labels_or_dummynode_ids = torch.cat(edge_labels_or_dummynode_ids, dim=0)

            graph_idx = node_embeddings, edge_matrix, edge_labels_or_dummynode_ids
            updated_nodes = self.forward(graph_idx)
            result = updated_nodes[tgt_idx]

            del updated_nodes, node_embeddings, edge_matrix, edge_labels_or_dummynode_ids, gathered
            torch.cuda.empty_cache()

            return result, []

    def encode_with_una(self, encoded, pre_encoded, una_idx, incremental=1, **kwargs):
        del kwargs

        assert not una_idx == pre_encoded is None

        if incremental == 0 or self.I < incremental:
            self.I += 1
            return encoded, pre_encoded, una_idx, []
        else:
            self.I = 1

            unseen_preencoded = set() if pre_encoded is None else set(range(pre_encoded.size(0)))
            for tgt_idx, una_emb in una_idx:
                encoded.append(torch.cat([pre_encoded[[tgt_idx]], una_emb], dim=-1))
                unseen_preencoded.discard(tgt_idx)
            assert not unseen_preencoded, (unseen_preencoded, [tgt for tgt, _ in una_idx])

            del pre_encoded, una_idx
            torch.cuda.empty_cache()

            return encoded, None, [], []
