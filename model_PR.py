import math
import numpy as np
import random
import torch.nn as nn
from transformers import BertModel


class Embeddings(nn.Module):
    """Token embedding layer: converts token indices to embeddings and scales by sqrt(d_model)."""
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)  # Embedding lookup table
        self.d_model = d_model

    def forward(self, x):
        # Scale embeddings for BERT
        return self.lut(x) * math.sqrt(self.d_model)


# BERT-based model for MIDI sequences with extended features
class MidiBert(nn.Module):
    """BERT model adapted for multi-feature MIDI input, including Pedal and P_Position."""
    def __init__(self, bertConfig):
        super().__init__()

        # Initialize BERT model
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # Token types and their embedding sizes
        self.n_tokens = [1262, 385, 89, 65, 767, 2, 385]  # Number of tokens per feature
        self.classes = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal', 'P_Position']
        self.emb_sizes = [128, 64, 32, 32, 128, 16, 64]

        # Embeddings for each token type
        self.word_emb = []
        for i in range(len(self.classes)):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # Linear layer to merge all feature embeddings into BERT input size
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

        # Optional dense layer for simpler embedding merging
        self.dense = nn.Linear(7, bertConfig.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        """Forward pass: map input features to embeddings and feed into BERT."""
        emb_linear = self.dense(input_ids.float())  # Merge features into BERT input dimension
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        return y

    def get_rand_tok(self):
        """Return a random velocity token (used for masking/randomization)."""
        vel_rand = random.choice(range(self.n_tokens[3]))
        return vel_rand


class MidiBertLM(nn.Module):
    """Language modeling head for MidiBert."""
    def __init__(self, midibert: MidiBert):
        super().__init__()
        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.hidden_size)  # Masked prediction head

    def forward(self, x, attn):
        # Get BERT outputs and apply MLM projection
        x = self.midibert(x, attn)
        return self.mask_lm(x)


class MLM(nn.Module):
    """Masked regression head predicting a single output feature (e.g., Pedal)."""
    def __init__(self, hidden_size):
        super().__init__()
        # Linear projection to one output dimension
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, y):
        # Use last hidden state of BERT
        y = y.hidden_states[-1]
        y1 = self.proj(y)
        return y1  # Return prediction tensor
