import tqdm
import random
from collections import defaultdict
import torch

from torch.nn.functional import log_softmax

import transformers

from graphmlp import positive_reinforcement_nllloss, GraphMLP
from util import reg_result, graph_result, combined_result
from evaluation import eval_combined

EXP_NEG_40 = torch.exp(torch.tensor(-40.))


def bhattacharyya_c(p, q, is_log=True):
    '''
    Computes the Bhattacharyya coefficient of (log) softmax distributions p and q.
    '''
    assert p.size() == q.size()
    p = torch.clamp_min(p, min=-40 if is_log else EXP_NEG_40)
    q = torch.clamp_min(q, min=-40 if is_log else EXP_NEG_40)
    return torch.nansum(torch.sqrt(torch.exp(p + q) if is_log else torch.mul(p, q)), dim=-1)


def diff(p, q, is_log=True):
    '''
    Computes the simple difference of (log) softmax distributions p and q.
    '''
    assert p.size() == q.size()
    p = torch.clamp_min(p, min=-40 if is_log else EXP_NEG_40)
    q = torch.clamp_min(q, min=-40 if is_log else EXP_NEG_40)
    return torch.nansum(torch.abs(torch.exp(p) - torch.exp(q) if is_log else p - q), dim=-1)


def log_dist(coeff):
    return lambda *args, **kwargs: -torch.log(coeff(*args, **kwargs))

bhattacharyya_d = log_dist(bhattacharyya_c)
diff_d = log_dist(diff)


class RegularizedLM(torch.nn.Module):
    def __init__(self, pretrained_lm: transformers.GPT2LMHeadModel, graph_model: GraphMLP):
        super(RegularizedLM, self).__init__()
        self.pretrained_lm = pretrained_lm
        self.graph_model = graph_model
        self.vocab_size = self.pretrained_lm.config.vocab_size

    def forward(self, plm_inputs, gm_inputs=None, plm_gm_tokens=None, loss_fxn=positive_reinforcement_nllloss(), lbda=.5,
                eps=.5, verbose=0, **kwargs):
        del kwargs
        plm_inputs = plm_inputs.clone()
        plm_labels = plm_inputs.clone()
        plm_inputs[plm_inputs == -100] = 0
        plm_outputs = self.pretrained_lm(plm_inputs, labels=plm_labels)
        lm_loss = plm_outputs.loss
        lm_logits = log_softmax(plm_outputs.logits, dim=-1)

        asserts = [not lm_loss.isnan(), lm_loss >= 0]
        assert_prints = [f'lm_loss == {lm_loss}']
        assert plm_outputs.logits.size(-1) == self.vocab_size
        loss = lm_loss.clone()

        if gm_inputs is not None:
            gm_labels = torch.gather(plm_labels, 1, plm_gm_tokens)
            gm_labels[gm_labels == self.graph_model.tokenizer.eos_token_id] = -100
            gm_labels = gm_labels.view(-1)
            gm_mask = gm_labels != -100
            gm_inputs = gm_inputs.view(-1, gm_inputs.size(-1))[gm_mask]
            gm_labels = gm_labels[gm_mask]
            gm_outputs = self.graph_model(gm_inputs,
                                          labels=gm_labels,
                                          softmax=False, loss_fxn=loss_fxn)
            gm_loss = gm_outputs.loss
            asserts.append(not gm_loss.isnan())
            asserts.append(gm_loss >= 0)
            assert_prints.append(f'gm_loss == {gm_loss}')
            assert gm_outputs.logits.size(-1) == self.vocab_size
            q = log_softmax(gm_outputs.logits, dim=-1)

            p = torch.gather(lm_logits, 1, torch.clamp_min(
                plm_gm_tokens.unsqueeze(-1).expand(-1, -1, self.vocab_size) - 1, min=0))
            p = p.view(-1, self.vocab_size)[gm_mask]

            p_not_nan = not torch.any(p.isnan())
            q_not_nan = not torch.any(q.isnan())
            asserts.append(p_not_nan)
            asserts.append(q_not_nan)
            assert_prints.append(f'p is {"not " if p_not_nan else ""}nan')
            assert_prints.append(f'q is {"not " if q_not_nan else ""}nan')

            pr_loss = torch.mean(bhattacharyya_d(p, q, is_log=True), dim=0)

            # # Trick 1: no regularization for ground truth answer
            # all_but_labels = torch.ones_like(p).scatter_(1, gm_labels.unsqueeze(1), 0.).bool()
            # _p = p[all_but_labels]
            # _q = q[all_but_labels]

            # pr_loss = torch.mean(bhattacharyya_d(_p, _q, is_log=True), dim=0)

            # # Trick 2: control regularization strength / direction
            # with torch.no_grad():
            #     #TODO: do this with log probs or exp probs?
            #     _p_q = _p + eps * _q

            # pr_loss = torch.mean(bhattacharyya_d(_p, _p_q, is_log=True), dim=0)

            # Trick 3: make regularization contingent on expected improvement
            # with torch.no_grad():
            #     per_word_ppl_loss = torch.nn.NLLLoss(reduction='none')
            #     p_ppls = per_word_ppl_loss(
            #         lm_logits[:, :-1].reshape(-1, self.vocab_size),
            #         plm_labels[:, 1:].reshape(-1))
            #     p_ppls = torch.gather(torch.cat(
            #         [torch.zeros(plm_gm_tokens.size(0), 1).to(p_ppls), p_ppls.view(plm_gm_tokens.size(0), -1)],
            #         dim=1), 1, plm_gm_tokens).view(-1)[gm_mask]
            #     q_ppls = per_word_ppl_loss(q, gm_labels)
            #     p_q_ppl_diff = p_ppls - q_ppls
            #     p_worse_mask = p_q_ppl_diff > 0
            #     q_worse_mask = p_q_ppl_diff < 0
            #     if verbose:
            #         p_better_ratio = torch.sum(q_worse_mask) / torch.sum(gm_mask)
            #         k = verbose
            #         top_k_p_better = [(self.pretrained_lm.tokenizer.decode(gm_labels[i]), d.item()) for d, i in
            #                           zip(*torch.topk(p_q_ppl_diff, k, largest=False))]
            #         top_k_q_better = [(self.pretrained_lm.tokenizer.decode(gm_labels[i]), d.item()) for d, i in
            #                           zip(*torch.topk(p_q_ppl_diff, k, largest=True))]
            #         print('LM ppl', torch.mean(p_ppls), ', graph ppl', torch.mean(q_ppls))
            #         print('LM better ratio', p_better_ratio.item(), torch.sum(q_worse_mask).item(),
            #               torch.sum(p_worse_mask).item())
            #         print('LM ppl - graph ppl')
            #         for x in top_k_p_better:
            #             print(x)
            #         print('...')
            #         for x in top_k_q_better[::-1]:
            #             print(x)
            #         print()
            #
            # if p_worse_mask.sum() == 0:
            #     pr_loss = torch.tensor(0.).to(p)
            #
            # else:
            #
            #     with torch.no_grad():
            #         __q = q[p_worse_mask]
            #     # pr_p_loss = bhattacharyya_d(p[p_worse_mask], __q, is_log=True)
            #
            #     # with torch.no_grad():
            #     #     __p = p[q_worse_mask]
            #     # pr_q_loss = bhattacharyya_d(__p, q[q_worse_mask], is_log=True)
            #
            #     # pr_loss = torch.mean(torch.cat([pr_p_loss, pr_q_loss], dim=0), dim=0)
            #
            #     # Trick 4: use simple difference instead of Bhattacharrya distance
            #     # pr_loss = torch.mean(diff(p, q, is_log=True), dim=0)
            #     pr_loss = torch.mean(diff(p[p_worse_mask], __q, is_log=True), dim=0)
            #
            #     # Trick 5: use mean-squared error of logits (see https://arxiv.org/pdf/2105.08919.pdf)
            #     # lm_logits = plm_outputs.logits
            #     # p = torch.gather(lm_logits, 1, plm_gm_tokens.unsqueeze(-1).expand(-1, -1, self.vocab_size))
            #     # p = p.view(-1, self.vocab_size)[gm_mask]
            #     # q = gm_outputs.logits
            #     # pr_loss = torch.nn.functional.mse_loss(p, q)

            asserts.append(not pr_loss.isnan())
            asserts.append(not (lbda * pr_loss).isnan())
            asserts.append(pr_loss >= 0)
            assert_prints.append(f'pr_loss == {pr_loss}')
            assert_prints.append(f'lbda * pr_loss == {lbda} * {pr_loss} == {lbda * pr_loss}')

            loss += gm_loss + lbda * pr_loss

        asserts.append(not loss.isnan())
        asserts.append(loss >= 0)
        assert_prints.append(f'loss == {loss}')
        assert all(asserts), (asserts, assert_prints)

        return reg_result(logits=torch.gather(plm_outputs.logits, 1, torch.clamp_min(
                                    plm_gm_tokens.unsqueeze(-1).expand(-1, -1, self.vocab_size) - 1, min=0)),
                          lm_result=plm_outputs,
                          graph_result=(gm_outputs if gm_inputs is not None \
                                            else graph_result(loss=torch.zeros(1),
                                                              logits=torch.zeros(1, self.vocab_size),
                                                              last_hidden_state=torch.zeros(1),
                                                              cluster_size=torch.zeros(1),
                                                              cluster_ratio=torch.zeros(1))),
                          aux_loss=pr_loss if gm_inputs is not None else torch.zeros(1),
                          aux_losses=(pr_loss,) if gm_inputs is not None else (),
                          loss=loss)


class CombinedLM(torch.nn.Module):
    def __init__(self, pretrained_lm: transformers.GPT2LMHeadModel, graph_model: GraphMLP, h_dim=768, dropout=0.2,
                 use_lm=True, use_graph=True):
        super(CombinedLM, self).__init__()
        self.pretrained_lm = pretrained_lm if use_lm else None
        self.graph_model = graph_model if use_graph else None
        self.tokenizer = graph_model.tokenizer
        self.vocab_size = pretrained_lm.config.vocab_size

        self.use_lm = use_lm
        self.use_graph = use_graph

    def forward(self, plm_inputs, gm_inputs, plm_gm_tokens, loss_fxn=positive_reinforcement_nllloss(),
                softmax=True, lm_weight=0., gm_weight=0., aux_weight=0., **kwargs):
        del kwargs

        plm_inputs = plm_inputs.clone()
        plm_labels = plm_inputs.clone()
        plm_inputs[plm_inputs == -100] = 0

        asserts = []
        assert_prints = []

        gm_labels = torch.gather(plm_labels, 1, plm_gm_tokens)
        gm_labels[gm_labels == self.tokenizer.eos_token_id] = -100
        gm_labels = gm_labels.view(-1)
        gm_mask = gm_labels != -100
        gm_inputs = gm_inputs.view(-1, gm_inputs.size(-1))[gm_mask]
        gm_labels = gm_labels[gm_mask]

        all_logits = []

        sub_losses = 0
        gm_outputs = None
        plm_outputs = None

        if self.use_graph:
            gm_outputs = self.graph_model(gm_inputs, labels=gm_labels, softmax=False, loss_fxn=loss_fxn)

            gm_loss = gm_outputs.loss
            sub_losses += gm_weight * gm_loss.clone()

            asserts.append(not gm_loss.isnan())
            asserts.append(gm_loss >= 0)
            assert_prints.append(f'gm_loss == {gm_loss}')
            assert gm_outputs.logits.size(-1) == self.vocab_size

            gm_logits = gm_outputs.logits
            all_logits.append(gm_logits)

        if self.use_lm:
            plm_outputs = self.pretrained_lm(plm_inputs, labels=plm_labels)

            lm_logits = torch.gather(plm_outputs.logits, 1, torch.clamp_min(
                plm_gm_tokens.unsqueeze(-1).expand(-1, -1, self.vocab_size) - 1, min=0))
            lm_logits = lm_logits.view(-1, self.vocab_size)[gm_mask]
            all_logits.append(lm_logits)

            lm_loss = plm_outputs.loss
            sub_losses += lm_weight * lm_loss.clone()

            asserts.append(not lm_loss.isnan())
            asserts.append(lm_loss >= 0)
            assert_prints.append(f'lm_loss == {lm_loss}')
            assert plm_outputs.logits.size(-1) == self.vocab_size

        out = sum(all_logits)
        out_softmax = log_softmax(out, dim=-1)

        pr_loss = torch.mean(bhattacharyya_d(log_softmax(lm_logits, dim=-1), out_softmax, is_log=True), dim=0)

        loss1, loss2, cluster_sizes, cluster_ratios = loss_fxn(out_softmax, gm_labels)
        loss = loss1 + loss2
        cluster_size = cluster_sizes.sum().item()
        cluster_ratio = cluster_ratios.sum().item()

        total_loss = loss + sub_losses + aux_weight * pr_loss

        return combined_result(logits=out_softmax if softmax else out,
                               loss=total_loss,
                               aux_loss=pr_loss,
                               aux_losses=(pr_loss,),
                               last_hidden_state=out,
                               lm_result=(plm_outputs if self.use_lm \
                                            else graph_result(loss=torch.zeros(1),
                                                              logits=torch.zeros(gm_inputs.size(0), self.vocab_size).to(gm_inputs),
                                                              last_hidden_state=torch.zeros(1),
                                                              cluster_size=torch.zeros(1),
                                                              cluster_ratio=torch.zeros(1))),
                               graph_result=(gm_outputs if self.use_graph \
                                            else graph_result(loss=torch.zeros(1),
                                                              logits=torch.zeros(gm_inputs.size(0), self.vocab_size).to(gm_inputs),
                                                              last_hidden_state=torch.zeros(1),
                                                              cluster_size=torch.zeros(1),
                                                              cluster_ratio=torch.zeros(1)))
                               )


def train(model, data, dev_data=None, n_data=None, randomize=True, train_mode=3, checkpoint_name='checkpoint.pt',
          seed=42, epochs=50, plm_lr=1e-5, gm_lr=1e-4, lr=1e-4,
          loss_fxn=positive_reinforcement_nllloss(), lbda=.5, lm_weight=0., gm_weight=0., aux_weight=1.):
    param_groups = []
    for module in model.children():
        params = list(module.parameters())
        if len(params) > 0 and any(p.requires_grad for p in params):
            if module == model.pretrained_lm:
                _lr = plm_lr
            elif module == model.graph_model:
                _lr = gm_lr
            else:
                _lr = lr
            param_groups.append({'params': params, 'lr': _lr})
    optim = torch.optim.AdamW(params=param_groups, weight_decay=.05)
    model.train()
    random.seed(seed)
    best_dev_ppl = float('inf')
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
                    for _, x_batch, l_batch, token_batch in pbar_batch:
                        optim.zero_grad()
                        model_outputs = model(l_batch, gm_inputs=x_batch, plm_gm_tokens=token_batch, loss_fxn=loss_fxn,
                                              lbda=lbda, lm_weight=lm_weight, gm_weight=gm_weight, aux_weight=aux_weight,
                                              c=0)
                                              # c=1 -> gold graphs only (teacher forcing)
                                              # c=0 -> auto graphs only
                        n += 1
                        loss += model_outputs.loss
                        batch_loss = model_outputs.loss  # / batch_n
                        batch_loss.backward()
                        optim.step()
                        pbar_batch.set_postfix(batch_loss=batch_loss.item(),
                                               lm_loss=model_outputs.lm_result.loss.item(),
                                               g_loss=model_outputs.graph_result.loss.item(),
                                               pr_loss=model_outputs.aux_loss.item())
                        pbar.set_postfix(total_loss=loss.item() / n)
                        if n_data is not None:
                            total_pbar.update(1 / n_data)

                if dev_data is not None:
                    model.eval()
                    domains = defaultdict(str)
                    dev_eval, _ = eval_combined(model, model.tokenizer, dev_data, domains, interesting_n=1,
                                                device=next(model.parameters()).device)
                    print()
                    for m in ('gold2', 'auto2', 'lm2'):
                        for p in ('ppl', 'acc'):
                            print(i, m, p, dev_eval['all'][m][p])

                    dev_ppl = dev_eval['all']['auto2' if train_mode >= 6 else 'gold2']['ppl']
                    if dev_ppl < best_dev_ppl:
                        best_dev_ppl = dev_ppl
                        print('saving checkpoint at epoch', i, 'with best ppl', dev_ppl)

                        if train_mode in (0, 2):
                            with open(checkpoint_name.replace('_model', '_graph_model'), 'wb') as f:
                                torch.save(model.graph_model.state_dict(), f)
                        if train_mode in (1, 2):
                            with open(checkpoint_name.replace('_model', '_lm_model'), 'wb') as f:
                                torch.save(model.pretrained_lm.state_dict(), f)
                        if train_mode in (3, 4, 5):
                            with open(checkpoint_name, 'wb') as f:
                                torch.save(model.state_dict(), f)
                        if train_mode in (6, 7):
                            with open(checkpoint_name, 'wb') as f:
                                torch.save(model.state_dict(), f)

                    model.train()
