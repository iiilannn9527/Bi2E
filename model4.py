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
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.dp = 0.2

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.gammas = nn.Embedding(nentity, 1)
        self.gammas.weight.data = torch.tensor([gamma]).repeat(nentity, 1)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Embedding(nentity, self.hidden_dim * 2)
        nn.init.uniform_(tensor=self.entity_embedding.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        self.relation_embedding = nn.Embedding(nrelation, hidden_dim * 2)
        nn.init.uniform_(tensor=self.relation_embedding.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        # self.entity_embedding_rota = nn.Embedding(nentity, hidden_dim * 6)
        # nn.init.uniform_(tensor=self.entity_embedding_rota.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        self.relation_embedding_bi = nn.Embedding(nrelation, hidden_dim * 4)
        nn.init.uniform_(tensor=self.relation_embedding_bi.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        # self.entity_location = nn.Embedding(nentity, self.hidden_dim)
        # nn.init.uniform_(tensor=self.entity_location.weight, a=0., b=2.)
        # self.er_fc = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(hidden_dim * 8, hidden_dim)
        # )
        # self.er_fc_1 = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(hidden_dim * 8, hidden_dim)
        # )
        # self.os_fc_1 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 4),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim * 4, hidden_dim)
        # )
        # self.os_fc_2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 4),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim * 4, hidden_dim)
        # )

        # self.alfa = nn.Parameter(torch.Tensor([0.001 * gamma / hidden_dim]), requires_grad=True)
        # self.alfas = nn.Embedding()
        """if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
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
            head = sample[:, 0]
            tail = sample[:, 2]

            # head = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=sample[:, 0]
            # ).unsqueeze(1)
            #
            # relation = torch.index_select(
            #     self.relation_embedding,
            #     dim=0,
            #     index=sample[:, 1]
            # ).unsqueeze(1)
            #
            # tail = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=sample[:, 2]
            # ).unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            relation_id = tail_part[:, 1]
            head = head_part.view(-1)
            tail = tail_part[:, 2]

            # head = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=head_part.view(-1)
            # ).view(batch_size, negative_sample_size, -1)
            #
            # relation = torch.index_select(
            #     self.relation_embedding,
            #     dim=0,
            #     index=tail_part[:, 1]
            # ).unsqueeze(1)
            #
            # tail = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=tail_part[:, 2]
            # ).unsqueeze(1)
        elif mode == 'tail-batch':
            # positive, negative
            head_part, tail_part = sample

            relation_id = head_part[:, 1]
            head = head_part[:, 0]
            tail = tail_part.view(-1)
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            #
            # # 嵌入
            # head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            # relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            # tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
            #                                                                                        negative_sample_size,
            #                                                                                        -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        model_func = {
            'four-bi': self.four_bi,
            'single-bi': self.single_bi,
            'TransE-bi': self.TransE_bi,
            'TransE': self.TransE,
            'PairRE': self.PairRE,
            'RotatE': self.Rota,
            'pRotatE': self.pRotatE,
            'osb-os-single-sa': self.os_Rota_single_sa,
            'osb-os': self.os_Rota_bi_os,
            'osb-os-sa': self.os_Rota_bi_os_sa,
            'os-rota-bi-osb-os-sa': self.os_Rota_bi_sa,
            # 'os-rota-bi-for': self.os_Rota_bi_for,
            # 'os-rota-bi-rota': self.os_Rota_bi_rota,
            'os-rota-bi': self.os_Rota_bi,
            # 'os-rota-bi-saf': self.os_Rota_bi_saf,
            # 'os-rota': self.os_Rota,
            # 'os-pair': self.os_Pair,
            # 'os-pair-1': self.os_Pair_1,
            # 'os-pair-2': self.os_Pair_2,
            # 'pair': self.PairRE,
            # 'rota': self.Rota,
            # 'single': self.single_middle,
            # 'bi': self.bi_middle,
            # 'single+Rota': self.smr,
            # 'bi-Rota': self.bmr,
            # 'adapt-bi': self.adapt_bi,
            # 'adapt-rota': self.adapt_rota,
            # 'exp': self.exp,
            # 'bi-1': self.bi_middle_1,
            # 'bi-2': self.bi_middle_2
        }
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation_id, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def single_bi(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding(relation).unsqueeze(1)
        r1, r2 = torch.chunk(relation, 2, -1)

        score1 = e1 * r1 - e2
        score2 = e2 * r2 - e1

        score1 = self.gamma.item() - torch.norm(score1, 1, -1)
        score2 = self.gamma.item() - torch.norm(score2, 1, -1)

        return (score1 + score2) / 2

    def four_bi(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding_bi(relation).unsqueeze(1)
        ro, rs, r3, r4, rro, rrs, rr3, rr4 = torch.chunk(relation, 8, -1)
        e1_o, e1_s, e13, e14 = torch.chunk(e1, 4, -1)
        e2_o, e2_s, e23, e24 = torch.chunk(e2, 4, -1)

        score_o = e1_o * ro - e2_o
        score_s = e1_s * rs - e2_s
        score_3 = e13 * r3 - e23
        score_4 = e14 * r4 - e24
        score_o = self.gamma.item() - torch.norm(score_o, 1, -1)
        score_s = self.gamma.item() - torch.norm(score_s, 1, -1)
        score_3 = self.gamma.item() - torch.norm(score_3, 1, -1)
        score_4 = self.gamma.item() - torch.norm(score_4, 1, -1)

        score_o1 = e2_o * rro - e1_o
        score_s1 = e2_s * rrs - e1_s
        score_33 = e23 * rr3 - e13
        score44 = e24 * rr4 - e14
        score_o1 = self.gamma.item() - torch.norm(score_o1, 1, -1)
        score_s1 = self.gamma.item() - torch.norm(score_s1, 1, -1)
        score_33 = self.gamma.item() - torch.norm(score_33, 1, -1)
        score44 = self.gamma.item() - torch.norm(score44, 1, -1)

        return ((score_o + score_s + score_o1 + score_s1) / 4 + (score_3 + score_4 + score_33 + score44) / 4) / 2

    def os_Rota_single_sa(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding(relation).unsqueeze(1)
        ro, rs = torch.chunk(relation, 2, -1)
        e1_o, e1_s = torch.chunk(e1, 2, -1)
        e2_o, e2_s = torch.chunk(e2, 2, -1)
        gamma_head = self.gammas(head).squeeze(-1)

        score_o = e1_o * ro - e2_o
        score_s = e1_s * rs - e2_s
        score_o = gamma_head - torch.norm(score_o, 1, -1)
        score_s = gamma_head - torch.norm(score_s, 1, -1)

        return (score_o + score_s) / 2

    def os_Rota_bi_os(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding_bi(relation).unsqueeze(1)
        ro, rs, rro, rrs = torch.chunk(relation, 4, -1)
        e1_o, e1_s = torch.chunk(e1, 2, -1)
        e2_o, e2_s = torch.chunk(e2, 2, -1)

        score_o = e1_o * ro - e2_o
        score_s = e1_s * rs - e2_s
        score_o = self.gamma.item() - torch.norm(score_o, 1, -1)
        score_s = self.gamma.item() - torch.norm(score_s, 1, -1)

        score_o1 = e2_o * rro - e1_o
        score_s1 = e2_s * rrs - e1_s
        score_o1 = self.gamma.item() - torch.norm(score_o1, 1, -1)
        score_s1 = self.gamma.item() - torch.norm(score_s1, 1, -1)

        return (score_o + score_s + score_o1 + score_s1) / 4

    def os_Rota_bi_os_sa(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding_bi(relation).unsqueeze(1)
        ro, rs, rro, rrs = torch.chunk(relation, 4, -1)
        e1_o, e1_s = torch.chunk(e1, 2, -1)
        e2_o, e2_s = torch.chunk(e2, 2, -1)
        gamma_head = self.gammas(head).squeeze(-1)
        gamma_tail = self.gammas(tail).squeeze(-1)

        score_o = e1_o * ro - e2_o
        score_s = e1_s * rs - e2_s
        score_o = gamma_head - torch.norm(score_o, 1, -1)
        score_s = gamma_head - torch.norm(score_s, 1, -1)

        score_o1 = e2_o * rro - e1_o
        score_s1 = e2_s * rrs - e1_s
        score_o1 = gamma_tail - torch.norm(score_o1, 1, -1)
        score_s1 = gamma_tail - torch.norm(score_s1, 1, -1)

        return (score_o + score_s + score_o1 + score_s1) / 4
    # def os_Rota(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding_rota(head)
    #     e2 = self.entity_embedding_rota(tail)
    #     relation = self.relation_embedding(relation).unsqueeze(1)
    #     ro, rs = torch.chunk(relation, 2, -1)
    #     e1, e1_o, e1_s = torch.chunk(e1, 3, -1)
    #     e2, e2_o, e2_s = torch.chunk(e2, 3, -1)
    #
    #     score_o = e1 * e1_o * ro - e2 * e2_o
    #     score_s = e1 * e1_s * rs - e2 * e2_s
    #
    #     score_o = self.gamma.item() - torch.norm(score_o, 1, -1)
    #     score_s = self.gamma.item() - torch.norm(score_s, 1, -1)
    #
    #     return score_o / 2 + score_s / 2
    def os_Rota_bi(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding_bi(relation).unsqueeze(1)
        ro, rs, rro, rrs = torch.chunk(relation, 4, -1)
        e1, e1_o, e1_s = torch.chunk(e1, 3, -1)
        e2, e2_o, e2_s = torch.chunk(e2, 3, -1)

        score_o = e1 * e1_o * ro - e2 * e2_o
        score_s = e1 * e1_s * rs - e2 * e2_s
        score_o = self.gamma.item() - torch.norm(score_o, 1, -1)
        score_s = self.gamma.item() - torch.norm(score_s, 1, -1)

        score_o1 = e2 * e2_o * rro - e1 * e1_o
        score_s1 = e2 * e2_s * rrs - e1 * e1_s
        score_o1 = self.gamma.item() - torch.norm(score_o1, 1, -1)
        score_s1 = self.gamma.item() - torch.norm(score_s1, 1, -1)

        return (score_o + score_s + score_o1 + score_s1) / 4
    # def os_Rota_bi_for(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     ro, rs, rro, rrs = torch.chunk(relation, 4, -1)
    #     e1, e1_o, e1_s = torch.chunk(e1, 3, -1)
    #     e2, e2_o, e2_s = torch.chunk(e2, 3, -1)
    #
    #     e1 = F.normalize(e1, 2, -1)
    #     e2 = F.normalize(e2, 2, -1)
    #
    #     score_o = e1 * e1_o * ro - e2 * e2_o
    #     score_s = e1 * e1_s * rs - e2 * e2_s
    #     score_o = self.gamma.item() - torch.norm(score_o, 1, -1)
    #     score_s = self.gamma.item() - torch.norm(score_s, 1, -1)
    #
    #     score_o1 = e2 * e2_o * rro - e1 * e1_o
    #     score_s1 = e2 * e2_s * rrs - e1 * e1_s
    #     score_o1 = self.gamma.item() - torch.norm(score_o1, 1, -1)
    #     score_s1 = self.gamma.item() - torch.norm(score_s1, 1, -1)
    #
    #     return (score_o + score_s + score_o1 + score_s1) / 4

    # def os_Rota_bi_rota(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding_rota(head)
    #     e2 = self.entity_embedding_rota(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     ro, rs, rro, rrs = torch.chunk(relation, 4, -1)
    #     e1, e1_o, e1_s = torch.chunk(e1, 3, -1)
    #     e2, e2_o, e2_s = torch.chunk(e2, 3, -1)
    #
    #     score_o = self.RotatE(e1 * e1_o, ro, e2 * e2_o, mode)
    #     score_s = self.RotatE(e1 * e1_s, rs, e2 * e2_s, mode)
    #     # score_o = e1 * e1_o * ro - e2 * e2_o
    #     # score_s = e1 * e1_s * rs - e2 * e2_s
    #     # score_o = self.gamma.item() - torch.norm(score_o, 1, -1)
    #     # score_s = self.gamma.item() - torch.norm(score_s, 1, -1)
    #
    #     score_o1 = self.RotatE(e2 * e2_o, rro, e1 * e1_o, mode)
    #     score_s1 = self.RotatE(e2 * e2_s, rrs, e1 * e1_s, mode)
    #     # score_o1 = e2 * e2_o * rro - e1 * e1_o
    #     # score_s1 = e2 * e2_s * rrs - e1 * e1_s
    #     # score_o1 = self.gamma.item() - torch.norm(score_o1, 1, -1)
    #     # score_s1 = self.gamma.item() - torch.norm(score_s1, 1, -1)
    #
    #     return (score_o + score_s + score_o1 + score_s1) / 4
    def os_Rota_bi_sa(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding_bi(relation).unsqueeze(1)
        ro, rs, rro, rrs = torch.chunk(relation, 4, -1)
        e1, e1_o, e1_s = torch.chunk(e1, 3, -1)
        e2, e2_o, e2_s = torch.chunk(e2, 3, -1)
        gamma_head = self.gammas(head).squeeze(-1)
        gamma_tail = self.gammas(tail).squeeze(-1)

        score_o = e1 * e1_o * ro - e2 * e2_o
        score_s = e1 * e1_s * rs - e2 * e2_s
        score_o = gamma_head - torch.norm(score_o, 1, -1)
        score_s = gamma_head - torch.norm(score_s, 1, -1)

        score_o1 = e2 * e2_o * rro - e1 * e1_o
        score_s1 = e2 * e2_s * rrs - e1 * e1_s
        score_o1 = gamma_tail - torch.norm(score_o1, 1, -1)
        score_s1 = gamma_tail - torch.norm(score_s1, 1, -1)

        return (score_o + score_s + score_o1 + score_s1) / 4
    # def os_Rota_bi_saf(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     ro, rs, rro, rrs = torch.chunk(relation, 4, -1)
    #     e1, e1_o, e1_s = torch.chunk(e1, 3, -1)
    #     e2, e2_o, e2_s = torch.chunk(e2, 3, -1)
    #     gamma_head = self.gammas(head).squeeze(-1)
    #     gamma_tail = self.gammas(tail).squeeze(-1)
    #     e1 = F.normalize(e1, 2, -1)
    #     e2 = F.normalize(e2, 2, -1)
    #
    #     score_o = e1 * e1_o * ro - e2 * e2_o
    #     score_s = e1 * e1_s * rs - e2 * e2_s
    #     score_o = gamma_head - torch.norm(score_o, 1, -1)
    #     score_s = gamma_head - torch.norm(score_s, 1, -1)
    #
    #     score_o1 = e2 * e2_o * rro - e1 * e1_o
    #     score_s1 = e2 * e2_s * rrs - e1 * e1_s
    #     score_o1 = gamma_tail - torch.norm(score_o1, 1, -1)
    #     score_s1 = gamma_tail - torch.norm(score_s1, 1, -1)
    #
    #     return (score_o + score_s + score_o1 + score_s1) / 4

    # def os_Pair(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     ro1, ro2, rs1, rs2 = torch.chunk(relation, 4, -1)
    #
    #     e1 = F.normalize(e1, 2, -1)
    #     e2 = F.normalize(e2, 2, -1)
    #     # 1
    #     score_1 = e1 * ro1 - e2 * rs1
    #     score_2 = e1 * ro2 - e2 * rs2
    #     score_3 = e1 * (ro1 + ro2) - e2 * (rs1 + rs2)
    #
    #     score_1 = self.gamma.item() - torch.norm(score_1, 1, -1)
    #     score_2 = self.gamma.item() - torch.norm(score_2, 1, -1)
    #     score_3 = self.gamma.item() - torch.norm(score_3, 1, -1)
    #
    #     return (score_1 + score_2 + score_3) / 3
    #
    # def os_Pair_1(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     ro1, ro2, rs1, rs2 = torch.chunk(relation, 4, -1)
    #
    #     e1 = F.normalize(e1, 2, -1)
    #     e2 = F.normalize(e2, 2, -1)
    #     # 1
    #     score_1 = e1 * ro1 - e2 * rs1
    #     score_2 = e1 * ro2 - e2 * rs2
    #     # score_3 = e1 * (ro1 + ro2) - e2 * (rs1 + rs2)
    #
    #     score_1 = self.gamma.item() - torch.norm(score_1, 1, -1)
    #     score_2 = self.gamma.item() - torch.norm(score_2, 1, -1)
    #     # score_3 = self.gamma.item() - torch.norm(score_3, 1, -1)
    #
    #     return (score_1 + score_2) / 2
    #
    # def os_Pair_2(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     ro1, ro2, rs1, rs2 = torch.chunk(relation, 4, -1)
    #
    #     e1 = F.normalize(e1, 2, -1)
    #     e2 = F.normalize(e2, 2, -1)
    #
    #     # 1
    #     score_1 = self.os_fc_1(e1) * self.os_fc_1(ro1) - self.os_fc_1(e2) * self.os_fc_1(rs1)
    #     score_2 = self.os_fc_2(e1) * self.os_fc_2(ro2) - self.os_fc_2(e2) * self.os_fc_2(rs2)
    #     # score_3 = e1 * (ro1 + ro2) - e2 * (rs1 + rs2)
    #
    #     score_1 = self.gamma.item() - torch.norm(score_1, 1, -1)
    #     score_2 = self.gamma.item() - torch.norm(score_2, 1, -1)
    #     # score_3 = self.gamma.item() - torch.norm(score_3, 1, -1)
    #
    #     return (score_1 + score_2) / 2
    #
    #
    # def bmr(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding_rota(head)
    #     e2 = self.entity_embedding_rota(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     r1, r2, r3, r4 = torch.chunk(relation, 4, -1)
    #
    #     # score = e1 * r1 - (e2 - self.alfa)
    #     # score = self.gamma.item() - torch.norm(score, 1, -1)
    #     # score1 = e1 * r2 - (e2 + self.alfa)
    #     # score1 = self.gamma.item() - torch.norm(score1, 1, -1)
    #     #
    #     # score_1 = e2 * r3 - (e1 - self.alfa)
    #     # score_1 = self.gamma.item() - torch.norm(score_1, 1, -1)
    #     # score_2 = e2 * r4 - (e1 + self.alfa)
    #     # score_2 = self.gamma.item() - torch.norm(score_2, 1, -1)
    #     score = self.RotatE(e1, r1, e2 - self.alfa, mode)
    #     score1 = self.RotatE(e1, r2, e2 + self.alfa, mode)
    #
    #     score_1 = self.RotatE(e2, r3, e1 - self.alfa, mode)
    #     score_2 = self.RotatE(e2, r4, e1 + self.alfa, mode)
    #
    #     return (score + score1 + score_1 + score_2) / 4
    #
    # def smr(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding_rota(head)
    #     e2 = self.entity_embedding_rota(tail)
    #     relation = self.relation_embedding(relation).unsqueeze(1)
    #     r1, r2 = torch.chunk(relation, 2, -1)
    #
    #     # score = e1 * r1 - (e2 - self.alfa)
    #     # score = self.gamma.item() - torch.norm(score, 1, -1)
    #     # score1 = e1 * r2 - (e2 + self.alfa)
    #     # score1 = self.gamma.item() - torch.norm(score1, 1, -1)
    #
    #     score = self.RotatE(e1, r1, e2 - self.alfa, mode)
    #     score1 = self.RotatE(e1, r2, e2 + self.alfa, mode)
    #
    #     return (score + score1) / 2
    #
    # def bi_middle(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     r1, r2, r3, r4 = torch.chunk(relation, 4, -1)
    #
    #     score = e1 * r1 - (e2 - self.alfa)
    #     score = self.gamma.item() - torch.norm(score, 1, -1)
    #     score1 = e1 * r2 - (e2 + self.alfa)
    #     score1 = self.gamma.item() - torch.norm(score1, 1, -1)
    #
    #     score_1 = e2 * r3 - (e1 - self.alfa)
    #     score_1 = self.gamma.item() - torch.norm(score_1, 1, -1)
    #     score_2 = e2 * r4 - (e1 + self.alfa)
    #     score_2 = self.gamma.item() - torch.norm(score_2, 1, -1)
    #     # score1 = e1 * r1 - e2
    #     # score1 = self.gamma.item() - torch.norm(score1, 1, -1)
    #     # # score2 = e2 * r2 - e1
    #     # # score2 = self.gamma.item() - torch.norm(score2, 1, -1)
    #
    #     return (score + score1 + score_1 + score_2) / 4
    #
    # def bi_middle_1(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     r1, r2, r3, r4 = torch.chunk(relation, 4, -1)
    #
    #     score_1 = e1 * (r1 + r2) / 2 - e2
    #     score_1 = self.gamma.item() - torch.norm(score_1, 1, -1)
    #
    #     score_2 = e2 * (r3 + r4) / 2 - e1
    #     score_2 = self.gamma.item() - torch.norm(score_2, 1, -1)
    #
    #     return (score_1 + score_2) / 2
    #
    # def bi_middle_2(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding_bi(relation).unsqueeze(1)
    #     r1, r2, r3, r4 = torch.chunk(relation, 4, -1)
    #
    #     score_1 = e1 * (r1 + r2) - e2
    #     score_1 = self.gamma.item() - torch.norm(score_1, 1, -1)
    #
    #     score_2 = e2 * (r3 + r4) - e1
    #     score_2 = self.gamma.item() - torch.norm(score_2, 1, -1)
    #
    #     return (score_1 + score_2) / 2
    #
    # def single_middle(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding(relation).unsqueeze(1)
    #     r1, r2 = torch.chunk(relation, 2, -1)
    #
    #     score = e1 * r1 - (e2 - self.alfa)
    #     score = self.gamma.item() - torch.norm(score, 1, -1)
    #     score1 = e1 * r2 - (e2 + self.alfa)
    #     score1 = self.gamma.item() - torch.norm(score1, 1, -1)
    #
    #     return (score + score1) / 2
    #
    # def er2e(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding(relation).unsqueeze(1)
    #     r1, r2 = torch.chunk(relation, 2, -1)
    #
    #     score1 = (e1 * r1 + e1 * r2) / 2 - e2
    #     score1 = self.gamma.item() - torch.norm(score1, 1, -1)
    #     # score2 = e2 * r2 - e1
    #     # score2 = self.gamma.item() - torch.norm(score2, 1, -1)
    #
    #     # score = score1 / 2 + score2 / 2
    #
    #     return score1
    #
    # def adapt_bi(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     r = self.relation_embedding(relation).unsqueeze(1)
    #     r1, r2 = torch.chunk(r, 2, -1)
    #
    #     # gamma = torch.index_select(self.gammas, dim=0, index=relation).unsqueeze(-1)
    #
    #     max_dim1 = max(e1.size(1), r1.size(1))
    #     if max_dim1 != 1:
    #         # e1 = e1.repeat(1, max_dim1//e1.size(1), 1)
    #         r1 = r1.repeat(1, max_dim1//r1.size(1), 1)
    #     er = torch.cat([e1, r1], dim=-1)
    #     er = self.er_fc(er)
    #     score = e1 * er - e2
    #     # score_in = torch.abs(score)
    #     # score_in = F.softmax(score_in * 1.0, dim=-1).detach() * score.size(-1)
    #     # score = score * score_in
    #     score = self.gamma.item() - torch.norm(score, 1, -1)
    #
    #     max_dim2 = max(e2.size(1), r2.size(1))
    #     if max_dim2 != 1:
    #         # e2 = e2.repeat(1, max_dim2 // e2.size(1), 1)
    #         r2 = r2.repeat(1, max_dim2 // r2.size(1), 1)
    #     er1 = torch.cat([e2, r2], dim=-1)
    #     er1 = self.er_fc_1(er1)
    #     score1 = e2 * er1 - e1
    #     # score_in_1 = torch.abs(score1)
    #     # score_in_1 = F.softmax(score_in_1 * 1.0, dim=-1).detach() * score1.size(-1)
    #     # score1 = score1 * score_in_1
    #     score1 = self.gamma.item() - torch.norm(score1, 1, -1)
    #
    #     return score / 2 + score1 / 2
    #
    # def adapt_rota(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding_rota(head)
    #     e2 = self.entity_embedding_rota(tail)
    #     r = self.relation_embedding(relation).unsqueeze(1)
    #     r1, r2 = torch.chunk(r, 2, -1)
    #
    #     max_dim1 = max(e1.size(1), r1.size(1))
    #     if max_dim1 != 1:
    #         # e1 = e1.repeat(1, max_dim1//e1.size(1), 1)
    #         r1 = r1.repeat(1, max_dim1//r1.size(1), 1)
    #     er = torch.cat([e1, r1], dim=-1)
    #     er = self.er_fc(er)
    #     # score = e1 * er - e2
    #     score = self.RotatE(e1, er, e2, mode)
    #
    #     max_dim2 = max(e2.size(1), r2.size(1))
    #     if max_dim2 != 1:
    #         # e2 = e2.repeat(1, max_dim2 // e2.size(1), 1)
    #         r2 = r2.repeat(1, max_dim2 // r2.size(1), 1)
    #     er1 = torch.cat([e2, r2], dim=-1)
    #     er1 = self.er_fc_1(er1)
    #     # score1 = e2 * er1 - e1
    #     score1 = self.RotatE(e2, er1, e1, mode)
    #
    #     return score / 2 + score1 / 2
    #
    # def exp(self, head, relation, tail, mode):
    #     head = head.view(relation.size(0), -1)
    #     tail = tail.view(relation.size(0), -1)
    #     e1 = self.entity_embedding(head)
    #     e2 = self.entity_embedding(tail)
    #     relation = self.relation_embedding(relation).unsqueeze(1)
    #     r1, r2 = torch.chunk(relation, 2, -1)
    #
    #     score = torch.pow(torch.exp(e1), r1) - torch.pow(torch.exp(e2), r2)
    #     score = self.gamma.item() - torch.norm(score, 1, -1)
    #
    #     return score
    #
    def PairRE(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding(relation).unsqueeze(1)
        r1, r2 = torch.chunk(relation, 2, -1)

        e1 = F.normalize(e1, 2, -1)
        e2 = F.normalize(e2, 2, -1)

        score = e1 * r1 - e2 * r2
        score = self.gamma.item() - torch.norm(score, 1, -1)

        return score

    def Rota(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        e1 = self.entity_embedding(head)
        e2 = self.entity_embedding(tail)
        relation = self.relation_embedding(relation).unsqueeze(1)
        r1, _ = torch.chunk(relation, 2, -1)

        score = self.RotatE(e1, r1, e2, mode)

        return score

    def RotatE(self, head, relation, tail, mode):
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
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        head = self.entity_embedding(head)
        tail = self.entity_embedding(tail)
        relation = self.relation_embedding(relation).unsqueeze(1)
        relation, r = torch.chunk(relation, 2, -1)

        pi = 3.14159262358979323846
        # Make phases of entities and relations uniformly distributed in [-pi, pi]
        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
            score1 = phase_tail + (r - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail
            score1 = (phase_tail + r) - phase_head

        score = torch.sin(score)
        score = torch.abs(score)
        score1 = torch.sin(score1)
        score1 = torch.abs(score1)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        score1 = self.gamma.item() - score1.sum(dim=2) * self.modules

        return (score + score1) / 2

    def TransE(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        head = self.entity_embedding(head)
        tail = self.entity_embedding(tail)
        relation = self.relation_embedding(relation).unsqueeze(1)
        relation, _ = torch.chunk(relation, 2, -1)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransE_bi(self, head, relation, tail, mode):
        head = head.view(relation.size(0), -1)
        tail = tail.view(relation.size(0), -1)
        head = self.entity_embedding(head)
        tail = self.entity_embedding(tail)
        relation = self.relation_embedding(relation).unsqueeze(1)
        r1, r2 = torch.chunk(relation, 2, -1)

        if mode == 'head-batch':
            score = head + (r1 - tail)
            score1 = tail + (r2 - head)
        else:
            score = (head + r1) - tail
            score1 = (tail + r2) - head

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        score1 = self.gamma.item() - torch.norm(score1, 1, 2)
        return (score + score1) / 2

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, writer, step):
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
        if step % 100 == 0:
            writer.add_scalar(f'Train/loss', loss * 100, step)
            writer.add_scalar(f'Train/positive loss', positive_sample_loss * 100, step)
            writer.add_scalar(f'Train/negative loss', negative_sample_loss * 100, step)

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
    def test_step(model, test_dataset, test_triples, all_true_triples, args, writer, test_step, id2type):
        '''
        Evaluate the model on test or valid datasets
        '''
        jump_boolean = False
        # if len(test_triples) > 1000:
        #     jump_boolean = True

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
            test_mode = ['tail-batch', 'head-batch']
            test_dataset_list = {'head-batch':test_dataloader_head, 'tail-batch':test_dataloader_tail}
            dataset_list = [test_dataloader_head, test_dataloader_tail]
            logs = []
            step = 0
            total_steps = sum([len(dataset) for dataset in dataset_list])
            with torch.no_grad():
                all_h1, all_h3, all_h10, all_mr, all_mrr = [], [], [], [], []
                for test_mode in test_mode:
                    if True:
                        rta = defaultdict(lambda: [])
                        rt1 = defaultdict(lambda: [])
                        rt3 = defaultdict(lambda: [])
                        rt10 = defaultdict(lambda: [])
                        rtr = defaultdict(lambda: [])

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
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset_list[test_mode]:
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

                        all_h10.append(H10)
                        all_h3.append(H3)
                        all_h1.append(H1)
                        all_mr.append(MR)
                        all_mrr.append(MRR)

                        writer.add_scalar(f'{test_dataset}/{test_mode}-H10', H10, test_step)
                        writer.add_scalar(f'{test_dataset}/{test_mode}-H3', H3, test_step)
                        writer.add_scalar(f'{test_dataset}/{test_mode}-H1', H1, test_step)
                        writer.add_scalar(f'{test_dataset}/{test_mode}-MR', MR, test_step)
                        writer.add_scalar(f'{test_dataset}/{test_mode}-MRR', MRR, test_step)

                        print(f'{mode} : H1 = {H1:.5f}, H3 = {H3:.5f}, H10 = {H10:.5f}, MR = {MR:.5f}, MRR = {MRR:.5f}')

                        if test_step % args.valid_steps == 0:
                            # print(f'{mode:^100}')
                            all_r = sorted(dict(r_type_count).items(), key=lambda x: x[0], reverse=False)
                            for i in range(len(all_r)):
                                all_r[i] = list(all_r[i])

                            for j in un_shoot.keys():
                                for i in range(len(all_r)):
                                    all_r[i].append(all_r[i][1] - un_shoot[j][all_r[i][0]])
                                    all_r[i].append(all_r[i][-1] / all_r[i][1])
                            if 1 == 0:
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
                                rta[id2type[i[0]]].append(i[1])
                                rt1[id2type[i[0]]].append(i[2])
                                rt3[id2type[i[0]]].append(i[4])
                                rt10[id2type[i[0]]].append(i[6])
                                rtr[id2type[i[0]]].extend(rank_all[i[0]])
                                if 1 == 0:
                                    print(
                                        "{0:^10}{1:^15}{2:^7}{3:^10.1f}{4:^7}{5:^10.1f}{6:^7}{7:^10.1f}{8:^7}{9:^10.1f}{10:^10.1f}{11:^10.5f}".format(
                                            i[0], i[1], i[2], i[3] * 100, i[4], i[5] * 100, i[6], i[7] * 100, i[8],
                                                              i[9] * 100,
                                            np.mean(rank_all[i[0]]), np.mean(1. / np.array(rank_all[i[0]]))))
                            for key in list(rta.keys()):
                                print("{0:^10}{1:^15}{2:^7}{3:^10.1f}{4:^7}{5:^10.1f}{6:^7}{7:^10.1f}{8:^10.1f}{9:^10.5f}".format(
                                        key, sum(rta[key]), sum(rt1[key]), sum(rt1[key])/sum(rta[key]) * 100,
                                        sum(rt3[key]), sum(rt3[key])/sum(rta[key]) * 100,
                                        sum(rt10[key]), sum(rt10[key])/sum(rta[key]) * 100,
                                        np.mean(rtr[key]), np.mean(1. / np.array(rtr[key]))))

                h10 = np.mean(all_h10)
                h3 = np.mean(all_h3)
                h1 = np.mean(all_h1)
                mr = np.mean(all_mr)
                mrr = np.mean(all_mrr)
                print(f'{test_step//100:0>4} total : H1 = {h1:.5f}, H3 = {h3:.5f}, H10 = {h10:.5f}, MR = {mr:.5f}, MRR = {mrr:.5f}')
                print('*' * 80)
                writer.add_scalar(f'{test_dataset}/H10', h10, test_step)
                writer.add_scalar(f'{test_dataset}/H3', h3, test_step)
                writer.add_scalar(f'{test_dataset}/H1', h1, test_step)
                writer.add_scalar(f'{test_dataset}/MR', mr, test_step)
                writer.add_scalar(f'{test_dataset}/MRR', mrr, test_step)

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
