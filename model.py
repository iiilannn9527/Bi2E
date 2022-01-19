#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from math import sqrt
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=True, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # self.rnn_layer = nn.RNN(input_size=200, hidden_size=200, num_layers=1)
        self.fc_layer = nn.Linear(200, 200, bias=False)

        """if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'PairRE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name == 'PairRE' and (not double_relation_embedding):
            raise ValueError('PairRE should use --double_relation_embedding')"""

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            relation_id = sample[:, 1]

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            relation_id = tail_part[:, 1]

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            # positive, negative
            head_part, tail_part = sample
            relation_id = head_part[:, 1]
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            # 嵌入
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            # 'TransE': self.TransE,
            # 'DistMult': self.DistMult,
            # 'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            # 'pRotatE': self.pRotatE,
            'PairRE': self.is_a,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode, relation_id)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    """def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score"""

    # 原版
    def PairRE_0(self, head, relation, tail, mode, relation_id):

        r1, r2 = torch.chunk(relation, 2, dim=-1)

        head = torch.sigmoid(self.fc_layer(head))
        tail = torch.sigmoid(self.fc_layer(tail))

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * r1 - tail * r2
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)

        return score

    # def PairRE(self, head, relation, tail, mode):
    #     re_head, re_tail = torch.chunk(relation, 2, dim=2)
    #     # r = self.relation_EM(relation)
    #     # r = r.view(r.size(0), self.nrelation, -1)
    #     #
    #     head = F.normalize(head, 2, -1)
    #     tail = F.normalize(tail, 2, -1)
    #
    #     # score = (head - tail) * re_head + (head - tail) * re_tail
    #     # score = (head - tail) * re_head - (head + tail) * re_tail
    #     score = head * re_head - tail * re_tail
    #     score = self.gamma.item() - torch.norm(score, p=1, dim=2)
    #     return score

    def rnn(self, head, relation, tail, mode, relation_id):
        r1, r2 = torch.chunk(relation, 2, -1)

        if r1.size(1) < head.size(1):
            r1 = r1.repeat(1, head.size(1), 1)
        if r2.size(1) < tail.size(1):
            r2 = r2.repeat(1, tail.size(1), 1)

        head_re = torch.stack([head, r1], dim=0)  # (2, b, N, hd)
        head_re = head_re.view(head_re.size(0), -1, head_re.size(-1))
        _, ht = self.rnn_layer(head_re)  # (1, b*N, hd)
        ht = ht[-1].view(head.size(0), -1, head.size(-1))
        ht = F.normalize(ht, 2, -1)
        tail_norm = F.normalize(tail, 2, -1)
        score1 = ht - tail_norm
        score1 = self.gamma.item() - torch.norm(score1, 1, -1)

        tail_re = torch.stack([tail, r2], dim=0)  # (2, b, N, hd)
        tail_re = tail_re.view(tail_re.size(0), -1, tail_re.size(-1))
        _, tail_ht = self.rnn_layer(tail_re)  # (1, b*N, hd)
        tail_ht = tail_ht[-1].view(tail.size(0), -1, tail.size(-1))
        tail_ht = F.normalize(tail_ht, 2, -1)
        head_norm = F.normalize(head, 2, -1)
        score2 = tail_ht - head_norm
        score2 = self.gamma.item() - torch.norm(score2, 1, -1)

        return score1 / 2 + score2 / 2

    """def rnn_hr1r2(self, head, relation, tail, mode, relation_id):
        ro, rs = torch.chunk(relation, 2, -1)

        r1, r2 = ro, rs
        if ro.size(1) < head.size(1):
            r1 = ro.repeat(1, head.size(1), 1)
            r2 = rs.repeat(1, head.size(1), 1)
        head_re = torch.stack([head, r1, r2], dim=0)  # (2/3, b, N, hd)
        head_re = head_re.view(head_re.size(0), -1, head_re.size(-1))
        _, head_ht = self.rnn_layer(head_re)  # (1, b*N, hd)
        head_ht = head_ht[-1].view(head.size(0), -1, head.size(-1))
        head_ht = F.normalize(head_ht, 2, -1)
        tail_norm = F.normalize(tail, 2, -1)
        score1 = head_ht - tail_norm
        score1 = self.gamma.item() - torch.norm(score1, 1, -1)

        r1, r2 = rs, ro
        if rs.size(1) < tail.size(1):
            r1 = rs.repeat(1, tail.size(1), 1)
            r2 = ro.repeat(1, tail.size(1), 1)
        tail_re = torch.stack([tail, r1, r2], dim=0)  # (2/3, b, N, hd)
        tail_re = tail_re.view(tail_re.size(0), -1, tail_re.size(-1))
        _, tail_ht = self.rnn_layer(tail_re)  # (1, b*N, hd)
        tail_ht = tail_ht[-1].view(tail.size(0), -1, tail.size(-1))
        tail_ht = F.normalize(tail_ht, 2, -1)
        head_norm = F.normalize(head, 2, -1)
        score2 = tail_ht - head_norm
        score2 = self.gamma.item() - torch.norm(score2, 1, -1)

        return score1 / 2 + score2 / 2"""

    def PairRE(self, head, relation, tail, mode, relation_id):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        abs = torch.abs(head - tail)
        abs = F.normalize(abs, 2, dim=-1)
        abs_max = torch.max(abs, dim=-1).values
        self_adapt = 1. - abs / (abs_max.view(abs_max.size(0), -1, 1) + 0.1)
        score = (head * re_head - tail * re_tail) * self_adapt

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def is_a(self, head, relation, tail, mode, relation_id):
        ro, rs = torch.chunk(relation, 2, dim=-1)
        # head1 = F.normalize(head * r + tail, 2, -1)
        # tail1 = F.normalize(tail * rs + head, 2, -1)
        # score1 = head1 - tail1
        # score1 = self.gamma.item() - torch.norm(score1, 1, -1)
        head2 = F.normalize(head, 2, -1)
        tail2 = F.normalize(tail, 2, -1)
        score2 = tail2 * rs - head2
        score2 = self.gamma.item() - torch.norm(score2, 1, -1)

        head3 = F.normalize(head, 2, -1)
        tail3 = F.normalize(tail, 2, -1)
        score3 = head3 * ro - tail3
        score3 = self.gamma.item() - torch.norm(score3, 1, -1)

        return (score2 + score3) / 2

    """# (hr-t) as extractor
    def is_a_1(self, head, relation, tail, mode, relation_id):
        r, rs = torch.chunk(relation, 2, dim=-1)
        # head1 = F.normalize(head1, 2, -1)
        # tail2 = F.normalize(tail2, 2, -1)
        #
        # tail1 = F.normalize(tail, 2, -1)
        # head2 = F.normalize(head, 2, -1)

        abs = torch.abs(head - tail).detach()
        abs = F.normalize(abs, 2, -1)
        abs_max = torch.max(abs, dim=-1).values
        self_adapt = 1. + abs / (abs_max.unsqueeze(-1) + sqrt(1 / self.hidden_dim))
        head = head * self_adapt
        tail = tail * self_adapt

        head1 = head * r
        tail2 = tail * rs

        abs_head = torch.abs(head1 - tail).detach()
        abs_head = F.normalize(abs_head, 2, -1)
        abs_head_max = torch.max(abs_head, dim=-1).values
        self_head_adapt = 1. + abs_head / (abs_head_max.view(abs_head_max.size(0), -1, 1) + sqrt(1 / self.hidden_dim))

        score1 = (head1 - tail) * self_head_adapt
        score1 = self.gamma.item() - torch.norm(score1, 1, -1)
        # second
        abs_tail = torch.abs(tail2 - head).detach()
        abs_tail = F.normalize(abs_tail, 2, -1)
        abs_tail_max = torch.max(abs_tail, dim=-1).values
        self_tail_adapt = 1. + abs_tail / (abs_tail_max.view(abs_tail_max.size(0), -1, 1) + sqrt(1 / self.hidden_dim))

        score2 = (tail2 - head) * self_tail_adapt
        score2 = self.gamma.item() - torch.norm(score2, 1, -1)
        # head2 = F.normalize(head * tail2, 2, -1)
        # tail2 = F.normalize(tail2, 2, -1)
        # score2 = tail2 - head2
        # score2 = self.gamma.item() - torch.norm(score2, 1, -1)

        return (score1 + score2) / 2

    # r fc extractor
    def is_a_2(self, head, relation, tail, mode, relation_id):
        r, rs = torch.chunk(relation, 2, dim=-1)
        er = self.er_fc(relation_id)
        er = er.view(er.size(0), -1, self.hidden_dim)
        assert er.size(1) == self.hidden_dim * 2

        if r.size(1) < head.size(1):
            r = r.repeat(1, head.size(1), 1)
        head1 = torch.cat([head, r], dim=-1)
        head1 = torch.matmul(head1, er)
        head1 = F.normalize(head1, 2, -1)
        # head1 = F.normalize(head * r, 2, -1)
        tail1 = F.normalize(tail, 2, -1)
        score1 = head1 - tail1
        score1 = self.gamma.item() - torch.norm(score1, 1, -1)

        if rs.size(1) < tail.size(1):
            rs = rs.repeat(1, tail.size(1), 1)
        tail2 = torch.cat([tail, rs], dim=-1)
        tail2 = torch.matmul(tail2, er)
        tail2 = F.normalize(tail2, 2, -1)
        head2 = F.normalize(head, 2, -1)
        # tail2 = F.normalize(tail * rs, 2, -1)
        score2 = tail2 - head2
        score2 = self.gamma.item() - torch.norm(score2, 1, -1)

        return (score1 + score2) / 2

    # fc(h cat r) relu
    def is_a_location(self, head, relation, tail, mode, relation_id):
        r_head, r_tail = torch.chunk(relation, 2, -1)

        if tail.size(1) < head.size(1):
            tail = tail.repeat(1, head.size(1), 1)
        elif tail.size(1) > head.size(1):
            head = head.repeat(1, tail.size(1), 1)

        # if r_tail.size(1) < tail.size(1):
        #     r_tail = r_tail.repeat(1, tail.size(1), 1)

        head_1 = F.normalize(head * r_head, 2, -1)
        tail_1 = F.normalize(tail, 2, -1)
        head_2 = F.normalize(head, 2, -1)
        tail_2 = F.normalize(tail * r_tail, 2, -1)

        location_head = torch.cat([head_1, tail_1], dim=-1)
        location_head = self.location(location_head)
        location_head = torch.where(location_head > 0., torch.full_like(location_head, 1.), location_head)

        score1 = (head_1 - tail_1) * location_head
        score1 = self.gamma.item() - torch.norm(score1, 1, -1)

        location_tail = torch.cat([tail_2, head_2], dim=-1)
        location_tail = self.location(location_tail)
        location_tail = torch.where(location_tail > 0., torch.full_like(location_tail, 1.), location_tail)
        score2 = (tail_2 - head_2) * location_tail
        score2 = self.gamma.item() - torch.norm(score2, 1, -1)

        return score1 / 2 + score2 / 2"""

    def RotatE(self, head, relation, tail, mode, relation_id):
        pi = 3.14159265358979323846
        # 一分为二
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = im_head * re_relation + re_head * im_relation
            # ???  距离应该用L1或者L2
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        # score.sum(dim = 2)来计算得分
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # 只计算了负采样的得分
        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            # 平均    1/k
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        jump_boolean = False
        if len(test_triples) > 10000:
            jump_boolean = True

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch'),
                batch_size=args.test_batch_size, num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn)
            test_dataloader_tail = DataLoader(
                TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch'),
                batch_size=args.test_batch_size, num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn)
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            logs = []
            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])
            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    if True:
                        r_type_count = defaultdict(lambda: 0)  # 总数计数
                        test_r = defaultdict(lambda: 0.)
                        h_shoot = {0: defaultdict(lambda: 0), 2: defaultdict(lambda: 0), 9: defaultdict(lambda: 0),
                                   49: defaultdict(lambda: 0)}
                        un_shoot = {0: defaultdict(lambda: 0), 2: defaultdict(lambda: 0), 9: defaultdict(lambda: 0),
                                    49: defaultdict(lambda: 0)}
                        rank_all = defaultdict(lambda: [])
                        hits = []
                        ranks = []
                        for i in range(50):
                            hits.append([])
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if True:
                            r_idx = positive_sample[:, 1]
                            for r in list(r_idx):
                                r_type_count[r.item()] += 1

                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)
                        # negative_sample  所有负例+（h,r,t)所对的t
                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            ranks.append(ranking)

                            if True:
                                rank_all[r_idx[i].item()].append(ranking)
                                for hits_level in h_shoot.keys():
                                    if ranking - 1 <= hits_level:
                                        hits[hits_level].append(1.0)
                                        h_shoot[hits_level][r_idx[i].item()] += 1
                                    else:
                                        hits[hits_level].append(0.0)
                                        un_shoot[hits_level][r_idx[i].item()] += 1
                                test_r[r_idx[i].item()] += 1

                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        if step > 1000 and jump_boolean:
                            step = 0
                            break

                        step += 1

                    if True:
                        H50 = np.mean(hits[49])
                        H10 = np.mean(hits[9])
                        H3 = np.mean(hits[2])
                        H1 = np.mean(hits[0])
                        MR = np.mean(ranks)
                        MRR = np.mean(1. / np.array(ranks))
                        print(
                            f'{mode} : H1 = {H1:.5f}, H3 = {H3:.5f}, H10 = {H10:.5f}, H50 = {H50:.5f}, MR = {MR:.5f}, MRR = {MRR:.5f}')

                        print(f'{mode:^100}')
                        all_r = sorted(dict(r_type_count).items(), key=lambda x: x[0], reverse=False)
                        for i in range(len(all_r)):
                            all_r[i] = list(all_r[i])

                        for j in un_shoot.keys():
                            for i in range(len(all_r)):
                                all_r[i].append(all_r[i][1] - un_shoot[j][all_r[i][0]])
                                all_r[i].append(all_r[i][-1] / all_r[i][1])
                        if 0 == 0:
                            print(
                                "{0:^10}{1:^15}{2:^7}{3:^10}{4:^7}{5:^10}{6:^7}{7:^10}{8:^7}{9:^10}{10:^10}{11:^10}".format(
                                    "relation", "total_number", "H1", "rate(%)", "H3", "rate(%)", "H10", "rate(%)",
                                    "H50", "rate(%)", "MR", "MRR"))
                            print(
                                "{0:^10}{1:^15}{2:^7}{3:^10.1f}{4:^7}{5:^10.1f}{6:^7}{7:^10.1f}{8:^7}{9:^10.1f}{10:^10.1f}{11:^10.5f}".format(
                                    "***", "***", "***", H1 * 100, "***", H3 * 100, "***", H10 * 100, "***", H50 * 100,
                                    MR, MRR))
                            print("{0:^100}".format(' ' * 30 + "-" * 20 + " " * 30))
                        for i in all_r:
                            if 0 == 0:
                                print(
                                    "{0:^10}{1:^15}{2:^7}{3:^10.1f}{4:^7}{5:^10.1f}{6:^7}{7:^10.1f}{8:^7}{9:^10.1f}{10:^10.1f}{11:^10.5f}".format(
                                        i[0], i[1], i[2], i[3] * 100, i[4], i[5] * 100, i[6], i[7] * 100, i[8],
                                                          i[9] * 100,
                                        np.mean(rank_all[i[0]]), np.mean(1. / np.array(rank_all[i[0]]))))

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
