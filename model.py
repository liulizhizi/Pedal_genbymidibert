import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertModel


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig):
        super().__init__()

        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types:
        self.n_tokens = [1262, 385, 89, 65, 767, 2, 385]
        self.classes = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal', 'P_Position']
        self.emb_sizes = [128, 64, 32, 32, 128, 16, 64]

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i in range(len(self.classes)):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

        self.dense = nn.Linear(7, bertConfig.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        emb_linear = self.dense(input_ids.float())
        # feed to bert
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        return y

    def get_rand_tok(self):
        vel_rand = random.choice(range(self.n_tokens[3]))
        return vel_rand


class MidiBertLM(nn.Module):
    def __init__(self, midibert: MidiBert):
        super().__init__()

        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.hidden_size)

    def forward(self, x, attn):
        x = self.midibert(x, attn)
        return self.mask_lm(x)


class MLM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)  # 输入维度调整

    def forward(self, y):  # 可选保留performer参数
        y = y.hidden_states[-1]
        y1 = self.proj(y)
        # y1 = self.duract(y1)
        return y1  # 直接返回张量或按需调整

