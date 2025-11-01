import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse earlier building blocks but add DecoderLayer with cross-attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def forward(self, x_q, x_kv, mask=None):
        # x_q: (B, T_q, d_model), x_kv: (B, T_k, d_model)
        b, t_q, _ = x_q.size()
        _, t_k, _ = x_kv.size()
        Q = self.W_q(x_q).view(b, t_q, self.h, self.d_k).transpose(1,2)  # (B,h,T_q,d_k)
        K = self.W_k(x_kv).view(b, t_k, self.h, self.d_k).transpose(1,2) # (B,h,T_k,d_k)
        V = self.W_v(x_kv).view(b, t_k, self.h, self.d_v).transpose(1,2) # (B,h,T_k,d_v)
        # if mask is not None:
        #     # mask shape should be broadcastable to (B, h, T_q, T_k)
        #     mask = mask.unsqueeze(1)
        out, attn = self.attn(Q, K, V, mask=mask)
        out = out.transpose(1,2).contiguous().view(b, t_q, self.h*self.d_v)
        return self.W_o(out), attn

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask=None):
        attn_out, attn = self.mha(x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # self-attention (causal)
        self_attn_out, self_attn = self.self_attn(x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        # cross-attention (attend to encoder output)
        cross_out, cross_attn = self.cross_attn(x, enc_out, mask=memory_mask)
        x = self.norm2(x + self.dropout(cross_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x, self_attn, cross_attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0).to(x.device)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, N=3, heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, src, src_mask=None):
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, src_mask=src_mask)
            attn_list.append(attn)
        x = self.norm(x)
        return x, attn_list

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, N=3, heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        self_attns = []
        cross_attns = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)
        x = self.norm(x)
        return x, self_attns, cross_attns

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, N=3, heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, d_model, N, heads, d_ff, dropout)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, N, heads, d_ff, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab)
    def make_pad_mask(self, seq, pad_idx=0):
        # seq: (B, T)
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T) -> broadcastable
    def make_causal_mask(self, size, device):
        # returns (1, 1, size, size) causal mask with 1s allowed and 0 masked
        mask = torch.tril(torch.ones((size, size), device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,size,size)
    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        src_mask = self.make_pad_mask(src, pad_idx=src_pad_idx)  # (B,1,1,T_src)
        tgt_pad_mask = self.make_pad_mask(tgt, pad_idx=tgt_pad_idx)  # (B,1,1,T_tgt)
        tgt_causal = self.make_causal_mask(tgt.size(1), tgt.device)  # (1,1,T_tgt,T_tgt)
        tgt_mask = tgt_pad_mask & tgt_causal  # broadcast to (B,1,T_tgt,T_tgt)
        memory_mask = src_mask  # (B,1,1,T_src) broadcastable to (B,1,T_tgt,T_src)
        memory, enc_attns = self.encoder(src, src_mask=src_mask)
        dec_out, self_attns, cross_attns = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.generator(dec_out)
        return logits, enc_attns, self_attns, cross_attns
