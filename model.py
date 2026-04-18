"""
model.py — MATCHA contrastive model architecture.

ContrastiveModel wraps a pretrained language model backbone and adds a
SenseNetwork that decomposes word embeddings into multiple "sense" vectors,
followed by a learned transformation and mean-pooling to produce a single
sentence embedding for contrastive learning.
"""

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN
from typing import Optional, Tuple


class ContrastiveModel(nn.Module):
    """Top-level model: backbone word embeddings -> SenseNetwork -> projection.

    Args:
        contxtl_model: Pretrained HuggingFace model used only for its embedding layer.
        config: SimpleNamespace with model_type, n_embd, num_senses, etc.
    """

    def __init__(self, contxtl_model, config):
        super().__init__()
        self.sense_network = SenseNetwork(config)
        self.contxtl_model = contxtl_model

        # Extract the word embedding layer from the backbone
        if config.model_type in ['gpt2', 'gpt_neo', 'roberta', 'xlm-roberta']:
            self.word_embeddings = self.contxtl_model.get_input_embeddings()
        elif config.model_type in ['mistral']:
            self.word_embeddings = self.contxtl_model.model.embed_tokens

        # Learnable transformation applied to sense vectors before pooling
        self.transformation_matrix = nn.Parameter(torch.randn(config.n_embd, config.n_embd))

    def get_model_output(self, input_ids):
        """Compute multi-sense embeddings from token IDs."""
        sense_input_embeds = self.word_embeddings(input_ids)  # (bs, s, d)
        senses = self.sense_network(sense_input_embeds)       # (bs, nv, s, d)
        return senses

    def forward(self, input_ids):
        """Produce a single sentence embedding by mean-pooling transformed senses.

        Returns:
            embedding: Tensor of shape (bs, d)
        """
        assert not torch.isnan(input_ids).any(), "Input IDs contain NaN values"

        senses = self.get_model_output(input_ids)              # (bs, nv, s, d)
        transformed_senses = senses @ self.transformation_matrix  # (bs, nv, s, d)
        embedding = transformed_senses.mean(dim=(1, 2))        # (bs, d)
        return embedding


class MLP(nn.Module):
    """Feed-forward block: linear -> activation -> linear -> dropout.

    Uses HuggingFace's Conv1D (equivalent to a linear layer applied
    along the last dimension) for compatibility with GPT-2 style configs.
    """

    def __init__(self, embed_dim, intermediate_dim, out_dim, config):
        super().__init__()
        self.c_fc = Conv1D(intermediate_dim, embed_dim)
        self.c_proj = Conv1D(out_dim, intermediate_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class NoMixBlock(nn.Module):
    """Transformer-style block *without* attention (no token mixing).

    Applies two residual sub-layers with layer normalization and dropout,
    where the only transformation is an MLP — tokens are processed independently.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config.n_embd, config.n_embd * 4, config.n_embd, config)
        self.resid_dropout1 = nn.Dropout(config.resid_pdrop)
        self.resid_dropout2 = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, residual):
        residual = self.resid_dropout1(hidden_states) + residual
        hidden_states = self.ln_1(residual)
        mlp_out = self.mlp(hidden_states)
        residual = self.resid_dropout2(mlp_out) + residual
        hidden_states = self.ln_2(residual)
        return hidden_states


class SenseNetwork(nn.Module):
    """Decomposes token embeddings into multiple sense vectors.

    Each token is mapped from (d,) to (num_senses, d) via a NoMixBlock
    followed by an MLP that expands the embedding dimension and reshapes.

    Input:  (bs, s, d)
    Output: (bs, num_senses, s, d)
    """

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.num_senses = config.num_senses
        self.n_embd = config.n_embd

        self.dropout = nn.Dropout(config.embd_pdrop)
        self.block = NoMixBlock(config)
        self.ln = nn.LayerNorm(self.n_embd, eps=config.layer_norm_epsilon)
        self.final_mlp = MLP(
            embed_dim=config.n_embd,
            intermediate_dim=config.sense_intermediate_scale * config.n_embd,
            out_dim=config.n_embd * config.num_senses,
            config=config,
        )

    def forward(self, input_embeds):
        residual = self.dropout(input_embeds)
        hidden_states = self.ln(residual)
        hidden_states = self.block(hidden_states, residual)
        senses = self.final_mlp(hidden_states)
        bs, s, nvd = senses.shape
        # Reshape from (bs, s, num_senses*d) -> (bs, num_senses, s, d)
        return senses.reshape(bs, s, self.num_senses, self.n_embd).transpose(1, 2)
