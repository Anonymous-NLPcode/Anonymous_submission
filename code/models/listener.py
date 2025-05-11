"""
Listener models
"""
import pdb

import torch
import torch.nn as nn
from . import rnn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence


class CopyListener(nn.Module):
    def __init__(self, feat_model, message_size=100, dropout=0.2):
        super().__init__()

        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.message_size = message_size
        self.fsize=16*64
        # self.flinear=nn.Linear(self.fsize,512)

        if self.message_size is None:
            self.bilinear = nn.Linear(self.feat_size, 1, bias=False)
        else:
            self.bilinear = nn.Linear(self.message_size, self.feat_size, bias=False)

    def embed_features(self, feats, edges, model_gcn_l, attention_l):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        parts_flat = feats.view(batch_size * n_obj, *rest)
        all_edges = torch.stack(edges)
        edgs_rest = all_edges.shape[2:]
        all_edges = all_edges.view(batch_size * n_obj, *edgs_rest)
        all_graph_feat = []
        for i in range(len(parts_flat)):
            node_fea = self.feat_model(parts_flat[i])
            gra_fea = model_gcn_l(node_fea, all_edges[i].long().cuda())
            # Global Mean Pooling
            # global_mean_pool = torch.mean(gra_fea, dim=0)
            # all_graph_feat.append(global_mean_pool)

            ## Global Attention Pooling
            attention_weights = attention_l(gra_fea)
            # 应用注意力权重
            weighted_features = attention_weights * gra_fea
            # 全局特征汇聚
            global_att_pool = torch.sum(weighted_features, dim=0).cuda()
            all_graph_feat.append(global_att_pool)
        all_graph_feat = torch.stack(all_graph_feat)
        all_graph_feat = all_graph_feat.unsqueeze(1).view(batch_size, n_obj, -1)
        feats_emb = all_graph_feat.unsqueeze(1).view(batch_size, n_obj, -1)
        feats_emb = self.dropout(feats_emb)

        # batch_size = feats.shape[0]
        # n_obj = feats.shape[1]
        # rest = feats.shape[2:]
        # feats_flat = feats.view(batch_size * n_obj, *rest)
        # feats_emb_flat = self.feat_model(feats_flat)
        #
        # feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        # feats_emb = self.dropout(feats_emb)

        return feats_emb

    def compare(self, feats_emb, message_enc):
        """
        Compute dot products
        """
        scores = torch.einsum("ijh,ih->ij", (feats_emb, message_enc))
        return scores

    def forward(self, feats, message):

        # Embed features
        # feats_emb = self.embed_features(feats)
        feats_emb=feats

        # Embed message
        if self.message_size is None:
            return self.bilinear(feats_emb).squeeze(2)
        else:
            message_bilinear = self.bilinear(message)

            return self.compare(feats_emb, message_bilinear)

    def reset_parameters(self):
        self.feat_model.reset_parameters()
        self.bilinear.reset_parameters()


class Listener(CopyListener):
    def __init__(self, feat_model, embedding_module, **kwargs):
        super().__init__(feat_model, **kwargs)

        self.embedding = embedding_module
        self.lang_model = rnn.RNNEncoder(self.embedding, hidden_size=self.message_size)
        self.vocab_size = embedding_module.num_embeddings

    def forward(self, parts_inp, edges, model_gcn_l, attention_l, lang, lang_length):
        # Embed features

        feats_emb = self.embed_features(parts_inp, edges, model_gcn_l, attention_l)  # 4×n×512
        # Embed language
        lang_emb = self.lang_model(lang, lang_length)
        # Bilinear term: lang embedding space -> feature embedding space
        new_lang = self.bilinear(lang_emb)    # 4 × 512 线性变换或全连接层来将特征向量的维度调整为一致的。

        return self.compare(feats_emb, new_lang), feats_emb

    def reset_parameters(self):
        super().reset_parameters()
        self.embedding.reset_parameters()
        self.lang_model.reset_parameters()


