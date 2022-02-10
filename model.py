import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, AdaptiveLogSoftmaxWithLoss
from adaptive import AdaptiveSoftmax, AdaptiveInput, AdaptiveTail


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5,
                 adsmax=False, adinp=False, tied_weights=False):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.adsmax = adsmax
        self.adinp = adinp
        self.tied_weights = tied_weights
        self.ninp = ninp
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        cutoffs = [round(ntoken / 15), 3 * round(ntoken / 15)]
        if self.tied_weights:
            if ninp != nhid:
                raise ValueError(f"Embedding size {ninp} should equal number "
                                 f"of FF hidden units {nhid} for tied weights.")
            shared_tail = AdaptiveTail(ninp, ntoken, cutoffs)
            self.adinp = True
            self.adsmax = True
            self.encoder = AdaptiveInput(ninp, ntoken, cutoffs, shared_tail=shared_tail)
            self.decoder = AdaptiveSoftmax(ninp, ntoken, cutoffs, shared_tail=shared_tail)
        else:
            if self.adinp:
                # self.encoder = AdaptiveInput(ninp, ntoken, cutoffs, shared_tail=None)
                self.encoder = AdaptiveInput(ntoken, -1, ninp, 4, ninp, cutoffs)
            else:
                self.encoder = nn.Embedding(ntoken, ninp)
                nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
            if self.adsmax:
                # self.decoder = AdaptiveSoftmax(ninp, ntoken, cutoffs, shared_tail=None)
                self.decoder = AdaptiveLogSoftmaxWithLoss(ninp, ntoken, cutoffs)
            else:
                self.decoder = nn.Linear(ninp, ntoken)
                nn.init.zeros_(self.decoder.weight)
                nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output
