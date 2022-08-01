import random
from collections import defaultdict
import tqdm

import torch

from graphmlp import GraphMLP
from evaluation import eval_combined
from util import cos_sim, get_batch_seq_mask


class SliceAutoEncoder(GraphMLP):
    def __init__(self, in_dim, h_dim1, h_dims, capacity_sizes, capacity_types, encoder, embedder, dropout=0.2):
        super(GraphMLP, self).__init__()
        self.encoder = encoder
        self.embedder = embedder
        self.hidden1 = torch.nn.Linear(in_dim, h_dim1)
        self.hiddens = torch.nn.ModuleList()
        last_hidden = h_dim1
        for h_dim in h_dims:
            self.hiddens.append(torch.nn.Linear(last_hidden, h_dim))
            last_hidden = h_dim
        self.last_hidden = last_hidden

        self.reverse_hiddens_residual = torch.nn.ModuleList()
        self.decoder_hiddens = torch.nn.ModuleList()
        for h in self.hiddens[::-1] + [self.hidden1]:
            o, i = h.weight.shape
            rhr = torch.nn.Linear(o, i)
            rhr.weight = torch.nn.Parameter(h.weight.t())
            self.reverse_hiddens_residual.append(rhr)
            self.decoder_hiddens.append(torch.nn.Linear(o, i))

        self.capacity_sizes = capacity_sizes
        self.capacity_types = capacity_types

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def _map_to_vocab(self, hidden):
        return hidden

    def encode_slice(self, x):
        result = self.forward(x, labels=None, softmax=False, loss_fxn=lambda x, y: (torch.nn.functional.nll_loss(x, y), 0, torch.zeros(1), torch.zeros(1)))
        return result.last_hidden_state

    def decode_hidden(self, h):
        x_prime_residual = h
        for decoder in self.reverse_hiddens_residual:
            x_prime_residual = decoder(x_prime_residual)

        x_prime_direct = h
        for decoder in self.decoder_hiddens:
            x_prime_direct = decoder(x_prime_direct)

        return (x_prime_residual + x_prime_direct) / 2

    def graph_snapshot_reconstruct_and_loss(self, struct_logits, embeddings, emb_mask,
                                            struct_tgt=None,
                                            use_tgt_for_mask=False,
                                            softmax_weight=1., sigmoid_weight=1., label_weight=1., emb_weight=1.):
        sub_logits = torch.split(struct_logits, self.capacity_sizes, dim=-1)
        softmaxes = []
        sigmoids = []
        # emb_softmaxes = []
        emb_sigmoids = []
        # embs = []
        reconstructed = []
        if struct_tgt is not None:
            sub_tgts = torch.split(struct_tgt, self.capacity_sizes, dim=-1)
            softmax_tgts = []
            sigmoid_tgts = []
            # emb_softmax_tgts = []
            emb_sigmoid_tgts = []
            # emb_tgts = []
        for i, (logits, typ) in enumerate(zip(sub_logits, self.capacity_types)):
            if struct_tgt is not None:
                tgts = sub_tgts[i]

            logits_sigmoid = torch.sigmoid(logits)

            if typ == 'hi_res_label':
                mask_threshold = tgts if use_tgt_for_mask and struct_tgt is not None else logits_sigmoid.detach()
                softmax_mask = torch.any(mask_threshold > 0.5, dim=-1)
                if struct_tgt is not None:
                    softmaxes.append(logits[softmax_mask])
                    softmax_tgt = torch.max(tgts, dim=-1).indices[softmax_mask]
                    softmax_tgts.append(softmax_tgt.detach())
                hardmaxes = torch.nn.functional.one_hot(torch.max(logits, dim=-1).indices, num_classes=logits.size(-1)).float()
                hardmaxes[~softmax_mask] = 0.
                reconstructed.append(hardmaxes)
                # TODO: maybe better to use expectation:
                # reconstructed.append(torch.softmax(logits, dim=-1))

                # also lossing sigmoids to get the right mask
                if struct_tgt is not None:
                    sigmoids.append(logits)
                    sigmoid_tgts.append(tgts)

            elif typ == 'lo_res_label':
                if struct_tgt is not None:
                    sigmoids.append(logits)
                    sigmoid_tgts.append(tgts)  # (tgts >= 0).float()
                reconstructed.append(logits_sigmoid)  # (torch.sigmoid(logits) > 0.5

            # elif typ == 'hi_res_emb':
            #     emb_sims = cos_sim(logits, embeddings) * emb_mask
            #
            #     if struct_tgt is not None:
            #         emb_tgt_sims = cos_sim(tgts, embeddings) * emb_mask  # TODO: would be faster to just get the token indices in index_graph(_fragment) and look up embeddings here and in forward
            #     softmax_mask = torch.any(emb_sims if struct_tgt is None else emb_tgt_sims > 0.5, dim=-1)
            #     if struct_tgt is not None:
            #         emb_softmaxes.append(emb_sims[softmax_mask])
            #         emb_softmax_tgt = torch.max(emb_tgt_sims, dim=-1).indices
            #         emb_softmax_tgts.append(emb_softmax_tgt[softmax_mask].detach())
            #     hardmaxes = torch.nn.functional.one_hot(torch.max(emb_sims, dim=-1).indices, num_classes=logits.size(-1))
            #     matched_embs = embeddings[hardmaxes]
            #     matched_embs[~softmax_mask] = 0.
            #     reconstructed.append(matched_embs)
            #
            #     # also lossing sigmoids to get the right mask
            #     if struct_tgt is not None:
            #         emb_sigmoids.append(emb_sims)
            #         emb_sigmoid_tgts.append(emb_tgt_sims)

            elif typ == 'emb':
                assert logits.size(1) == embeddings.size(1), (logits.size(), embeddings.size())
                emb_sims = cos_sim(logits, embeddings) * emb_mask

                if struct_tgt is not None:
                    emb_sigmoids.append(emb_sims)
                    emb_tgt_sims = cos_sim(tgts, embeddings) * emb_mask
                    emb_sigmoid_tgts.append(emb_tgt_sims.detach())
                weighted_embs = torch.matmul(torch.nn.functional.normalize(emb_sims, p=1, dim=-1), embeddings)
                reconstructed.append(weighted_embs)
                # embs.append(logits)
                # emb_tgts.append(tgts)
                # reconstructed.append(logits)

        if struct_tgt is not None:
            softmaxes = torch.cat(softmaxes, dim=0)
            sigmoids = torch.cat(sigmoids, dim=0)
            # emb_softmaxes = torch.cat(emb_softmaxes, dim=0)
            emb_sigmoids = torch.cat(emb_sigmoids, dim=0)
            # embs = torch.cat(embs, dim=0)
            softmax_tgts = torch.cat(softmax_tgts, dim=0)
            sigmoid_tgts = torch.cat(sigmoid_tgts, dim=0)
            # emb_softmax_tgts = torch.cat(emb_softmax_tgts, dim=0)
            emb_sigmoid_tgts = torch.cat(emb_sigmoid_tgts, dim=0)
            # emb_tgts = torch.cat(emb_tgts, dim=0)
        reconstructed = torch.cat(reconstructed, dim=-1)

        hi_res_loss = torch.zeros_like(struct_logits).mean()
        lo_res_loss = torch.zeros_like(struct_logits).mean()
        emb_loss = torch.zeros_like(struct_logits).mean()
        loss = torch.zeros_like(struct_logits).mean()
        if struct_tgt is not None:
            softmax_loss = torch.nn.CrossEntropyLoss()  # TODO: add token-level option for eval
            logit_sigmoid_loss = torch.nn.BCEWithLogitsLoss()
            sigmoid_loss = torch.nn.BCELoss()
            # cosine_loss = torch.nn.CosineEmbeddingLoss()
            assert torch.all(0 <= emb_sigmoids)
            assert torch.all(emb_sigmoids <= 1)
            hi_res_loss = softmax_loss(softmaxes, softmax_tgts)       #  * softmaxes[0].size(-1)
            lo_res_loss = logit_sigmoid_loss(sigmoids, sigmoid_tgts)  # * sigmoids[0].size(-1)
            emb_loss = sigmoid_loss(emb_sigmoids, emb_sigmoid_tgts)   # * emb_sigmoids[0].size(-1)
            # emb_loss = cosine_loss(embs, emb_tgts, torch.ones(embs.size(0)))
            loss = softmax_weight * label_weight * hi_res_loss + \
                   sigmoid_weight * label_weight * lo_res_loss + \
                   sigmoid_weight * emb_weight * emb_loss   #  * sigmoids[0].size(-1) #  * embs[0].size(-1)
                   # softmax_weight * emb_weight * softmax_loss(emb_softmaxes, emb_softmax_tgts) + \

        return loss, reconstructed, hi_res_loss, lo_res_loss, emb_loss


def train(model, data, dev_data=None, n_data=None, randomize=True, checkpoint_name='checkpoint.pt',
          seed=42, epochs=50, lr=1e-4):
    param_groups = []
    for module in model.children():
        params = list(module.parameters())
        if len(params) > 0 and any(p.requires_grad for p in params):
            param_groups.append({'params': params, 'lr': lr})
    optim = torch.optim.AdamW(params=param_groups, weight_decay=.05)
    model.train()
    random.seed(seed)
    best_dev_loss = float('inf')
    with tqdm.tqdm(None, total=epochs, desc=f'Total', unit_scale=True) as total_pbar:
        for i in range(epochs):
            with tqdm.tqdm(None, desc=f'Total - Epoch {i + 1}', total=epochs) as pbar:
                pbar.update(i + 1)
                total_pbar.set_description(f'Total - Epoch {i + 1}')
                loss = 0
                n = 0
                if randomize:
                    random.shuffle(data)
                with tqdm.tqdm(data, total=n_data, desc=f'Epoch {i + 1}') \
                        as pbar_batch:
                    for _, x_batch, l_batch, _ in pbar_batch:
                        optim.zero_grad()
                        encoded_hidden = model.encode_slice(x_batch)
                        decoded_slice = model.decode_hidden(encoded_hidden)
                        assert x_batch.shape == decoded_slice.shape
                        bs, sl, d = x_batch.shape
                        embeddings = model.embedder(l_batch)
                        emb_mask = get_batch_seq_mask(bs, sl - 1).view(bs * (sl - 1), -1)
                        batch_loss, reconstructed, hi_res_loss, lo_res_loss, emb_loss = \
                            model.graph_snapshot_reconstruct_and_loss(decoded_slice.view(bs*sl, d),
                                                                      embeddings.view(bs*sl, -1),
                                                                      emb_mask.view(-1),
                                                                      struct_tgt=x_batch.view(bs*sl, d).detach())
                        del x_batch
                        n += 1
                        loss += batch_loss.detach()
                        batch_loss.backward()
                        optim.step()
                        pbar_batch.set_postfix(batch_loss=batch_loss.item(),
                                               hi_res_loss=hi_res_loss.item(),
                                               lo_res_loss=lo_res_loss.item(),
                                               emb_loss=emb_loss.item())
                        pbar.set_postfix(total_loss=loss.item() / n, mem='{:.1f} MiB'.format(torch.cuda.max_memory_allocated() / 1000000))
                        if n_data is not None:
                            total_pbar.update(1 / n_data)
                        torch.cuda.empty_cache()

                if dev_data is not None:
                    model.eval()
                    dev_loss = 0
                    n = 0
                    for _, x_batch, _, _, _ in dev_data:
                        encoded_hidden = model.encode_slice(x_batch)
                        decoded_slice = model.decode_hidden(encoded_hidden)
                        batch_loss, reconstructed, hi_res_loss, lo_res_loss, emb_loss = \
                            model.graph_snapshot_reconstruct_and_loss(decoded_slice, struct_tgt=x_batch)
                        del x_batch
                        n += 1
                        dev_loss += batch_loss.detach()

                    dev_loss /= n
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        print('saving checkpoint at epoch', i, 'with best loss', dev_loss)

                        with open(checkpoint_name, 'wb') as f:
                            torch.save(model.state_dict(), f)

                    model.train()
