"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MultiheadAttention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from lib.models.aiatrack.attention import AiAModule


def check_inf(tensor):
    return torch.isinf(tensor.detach()).any()


def check_nan(tensor):
    return torch.isnan(tensor.detach()).any()


def check_valid(tensor, type_name):
    if check_inf(tensor):
        print('%s is inf' % type_name)
    if check_nan(tensor):
        print('%s is nan' % type_name)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False,
                 divide_norm=False, use_AiA=True, match_dim=64, feat_size=400):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                divide_norm=divide_norm, use_AiA=use_AiA,
                                                match_dim=match_dim, feat_size=feat_size)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                divide_norm=divide_norm, use_AiA=use_AiA,
                                                match_dim=match_dim, feat_size=feat_size)
        decoder_norm = nn.LayerNorm(d_model)
        if num_decoder_layers == 0:
            self.decoder = None
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def run_encoder(self, feat, mask, pos_emb, inr_emb):
        """
        Args:
            feat: (H1W1+H2W2, bs, C)
            mask: (bs, H1W1+H2W2)
            pos_embed: (H1W1+H2W2, bs, C)
        """

        return self.encoder(feat, src_key_padding_mask=mask, pos=pos_emb, inr=inr_emb)

    def run_decoder(self, search_mem, refer_mem_list, refer_emb_list, refer_pos_list, refer_msk_list):
        """
        Args:
            search_mem: (HW, bs, C)
            pos_emb: (HW, bs, C)
        """

        return self.decoder(search_mem, refer_mem_list, refer_emb_list, refer_pos_list, refer_msk_list)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # clone 3 copies
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                inr: Optional[Tensor] = None):
        output = src  # (HW,B,C)

        for stack, layer in enumerate(self.layers):
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos, inr=inr)

        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, refer_mem_list, refer_emb_list, refer_pos_list, refer_msk_list):
        output = tgt

        for stack, layer in enumerate(self.layers):
            output = layer(output, refer_mem_list, refer_emb_list, refer_pos_list, refer_msk_list)

        if self.norm is not None:
            output = self.norm(output)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False, divide_norm=False,
                 use_AiA=True, match_dim=64, feat_size=400):
        super().__init__()
        self.self_attn = AiAModule(d_model, nhead, dropout=dropout,
                                   use_AiA=use_AiA, match_dim=match_dim, feat_size=feat_size)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # First normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                inr: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # Add pos to src
        if self.divide_norm:
            # Encoder divide by norm
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        # src2 = self.self_attn(q, k, value=src)[0]
        src2 = self.self_attn(query=q, key=k, value=src, pos_emb=inr, key_padding_mask=src_key_padding_mask)[0]
        # Add and norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add and Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False, divide_norm=False,
                 use_AiA=True, match_dim=64, feat_size=400):
        super().__init__()
        self.long_term_attn = AiAModule(d_model, nhead, dropout=dropout,
                                        use_AiA=use_AiA, match_dim=match_dim, feat_size=feat_size)
        self.short_term_attn = AiAModule(d_model, nhead, dropout=dropout,
                                         use_AiA=use_AiA, match_dim=match_dim, feat_size=feat_size)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, refer_mem_list, refer_emb_list, refer_pos_list, refer_msk_list):
        # Mutual attention
        mem_ensemble = torch.cat(refer_mem_list[0:1], dim=0)
        emb_ensemble = torch.cat(refer_emb_list[0:1], dim=0)
        refer_pos = torch.cat(refer_pos_list[0:1], dim=0)
        refer_msk = torch.cat(refer_msk_list[0:1], dim=1)
        refer_queries = tgt
        refer_keys = mem_ensemble
        refer_values = mem_ensemble + emb_ensemble
        if self.divide_norm:
            refer_queries = refer_queries / torch.norm(refer_queries, dim=-1, keepdim=True) * self.scale_factor
            refer_keys = refer_keys / torch.norm(refer_keys, dim=-1, keepdim=True)
        long_tgt_refer, long_attn_refer = self.long_term_attn(query=refer_queries,
                                                              key=refer_keys,
                                                              value=refer_values,
                                                              pos_emb=refer_pos,
                                                              key_padding_mask=refer_msk)
        mem_ensemble = torch.cat(refer_mem_list[1:], dim=0)
        emb_ensemble = torch.cat(refer_emb_list[1:], dim=0)
        refer_pos = torch.cat(refer_pos_list[1:], dim=0)
        refer_msk = torch.cat(refer_msk_list[1:], dim=1)
        refer_queries = tgt
        refer_keys = mem_ensemble
        refer_values = mem_ensemble + emb_ensemble
        if self.divide_norm:
            refer_queries = refer_queries / torch.norm(refer_queries, dim=-1, keepdim=True) * self.scale_factor
            refer_keys = refer_keys / torch.norm(refer_keys, dim=-1, keepdim=True)
        short_tgt_refer, short_attn_refer = self.short_term_attn(query=refer_queries,
                                                                 key=refer_keys,
                                                                 value=refer_values,
                                                                 pos_emb=refer_pos,
                                                                 key_padding_mask=refer_msk)
        tgt = tgt + self.dropout1_1(long_tgt_refer) + self.dropout1_2(short_tgt_refer)
        tgt = self.norm1(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # Add and Norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
        num_decoder_layers=cfg.MODEL.TRANSFORMER.DEC_LAYERS,
        normalize_before=cfg.MODEL.TRANSFORMER.PRE_NORM,
        divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM,
        use_AiA=cfg.MODEL.AIA.USE_AIA,
        match_dim=cfg.MODEL.AIA.MATCH_DIM,
        feat_size=cfg.MODEL.AIA.FEAT_SIZE
    )


def _get_activation_fn(activation):
    """
    Return an activation function given a string.
    """

    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'ERROR: activation should be relu/gelu/glu, not {activation}')
