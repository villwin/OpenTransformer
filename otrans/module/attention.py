# File   : attention.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)
from torch.autograd import Variable

class BasedAttention(nn.Module):
    def __init__(self, source_dim, output_dim, enable_output_proj=True, dropout=0.0):
        super(BasedAttention, self).__init__()

        self.enable_output_proj = enable_output_proj
        if self.enable_output_proj:
            self.output_proj = nn.Linear(source_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def compute_context(self, values, scores, mask=None):
        """
        Args:
            values: [b, t2, v] or [b, nh, t2, v]
            scores: [b, t1, t2] or [b, nh, t1, t2]
            mask: [b, t1, t2] or [b, 1/nh, t1, t2]
        """

        assert values.dim() == scores.dim()

        if mask is not None:
            scores.masked_fill_(~mask, -float('inf'))
        
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, values)

        if context.dim() == 4:
            b, n, t, v = context.size()
            context = context.transpose(1, 2).reshape(b, t, n * v)
        
        if self.enable_output_proj:
            context = self.output_proj(context)

        return self.dropout(context), weights

def scale_offset(x):
    gamma = torch.var(x.shape[-1:])
    return x*gamma+gamma

def rope(x,axis):
    shape=x.size()
    if isinstance(axis,int):
        axis=[axis]
    spatial_shape=[shape[i] for i in axis]
    total_len =1
    for i in spatial_shape:
        total_len *=  i
    temp=torch.arange(start=0,end=total_len)
    #temp=temp.float32()
    position = torch.reshape(temp, spatial_shape)
    for i in range(axis[-1:][0]+ 1,len(shape)-1,1):
        position = torch.unsqueeze(position,dim=-1)
    half_size = shape[-1:][0] // 2
    temp = torch.arange(0,half_size)
    #temp= torch.float32
    freq_seq = temp / float(half_size)
    inv_freq = 10000 ** -freq_seq
    sinusoid = torch.einsum(' ...,d -> ...d ', position, inv_freq)
    sin = torch.sin(sinusoid).cuda()
    cos = torch.cos(sinusoid).cuda()
    #x1,x2 = torch.split(x,2,dim=-1)
    x1,x2=torch.chunk(x,2,dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin],dim=-1)
class Act_op(nn.Module):
    def __init__(self):
        super(Act_op,self).__init__()
    def forward(self,x):
        x=x*torch.sigmoid(x)
        return  x

class MultiHeadedSelfAttention(BasedAttention):
    def __init__(self, n_heads, d_model, dropout_rate=0.0, share_qvk_proj=False, rank_scale=0.5):
        super(MultiHeadedSelfAttention, self).__init__(int(d_model * rank_scale), d_model, enable_output_proj=True,
                                                       dropout=dropout_rate)

        self.d_model = d_model
        self.share_qvk_proj = share_qvk_proj
        self.nheads = n_heads
        self.d_k = int(d_model // n_heads * rank_scale)
        self.rank_scale = rank_scale
        # self.qvk_proj = nn.Linear(d_model, int(d_model) if self.share_qvk_proj else int(d_model * 3))
        self.qvk_proj = nn.Linear(d_model,
                                  int(d_model * rank_scale) if self.share_qvk_proj else int(d_model * 3 * rank_scale))

    def forward(self, x, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, int(self.d_model * self.rank_scale), dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights

    def inference(self, x, mask, cache=None):

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights, cache


class MultiHeadedCrossAttention(BasedAttention):
    def __init__(self, n_heads, d_model, memory_dim, dropout_rate=0.0, share_vk_proj=False):
        super(MultiHeadedCrossAttention, self).__init__(d_model, d_model, enable_output_proj=True, dropout=dropout_rate)

        self.d_model = d_model
        self.share_vk_proj = share_vk_proj
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.vk_proj = nn.Linear(memory_dim, d_model if self.share_vk_proj else d_model * 2)

    def forward(self, query, memory, memory_mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = torch.split(memory, self.d_model, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))

        return context, attn_weights

    def inference(self, query, memory, memory_mask, cache=None):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = torch.split(memory, self.d_model, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))

        return context, attn_weights, cache


class MultiHeadedSelfAttentionWithRelPos(BasedAttention):
    def __init__(self, n_heads, d_model, dropout_rate=0.0, skip_term_b=False, share_qvk_proj=False):
        super(MultiHeadedSelfAttentionWithRelPos, self).__init__(n_heads, d_model, dropout_rate, share_qvk_proj)
        self.s = 128
        self.gamma = nn.Parameter(torch.Tensor(2, self.s))
        self.beta = nn.Parameter(torch.Tensor(2, self.s))
        torch.nn.init.xavier_normal_(self.gamma)
        torch.nn.init.constant_(self.beta, 0)
        self.layernorm = nn.LayerNorm(d_model)
        self.silu = Act_op()
        expansion_factor = 1
        self.e = int(d_model * expansion_factor)
        self.dense = nn.Linear(d_model, 2 * self.e + self.s)
        self.x_proj = nn.Linear(self.e, d_model)
    def _mask_positions(self, qe):
        #QEr is a matrix of queries (absolute position) dot distance embeddings (relative pos).
        #Mask out invalid relative positions: e.g. if sequence length is L, the query at
        #L-1 can only attend to distance r = 0 (no looking backward).
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device='cuda'), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)
    def get_index(self, seq_len):
        index = math.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)
    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]
    def forward(self, x, mask, pos):
        b,t,_=x.size()
        m = max(b, t)
        shortcut, x = x, self.layernorm(x)
        s = 128
        uv = self.dense(x)
        u, v, base = torch.split(self.silu(uv), [self.e, self.e, s],dim=-1)
        # Generate Query (q) and Key (k) from base.
        base = torch.einsum(' ...r,hr -> ...hr ', base, self.gamma) + self.beta
        base = rope(base, axis=1)
        q, k = torch.chunk(base,2, dim=-2)
        weight_index = self.get_index(m).to(q)
        b,c,n,dk=q.size()
        q=q.contiguous().transpose(1,2)
        k = k.contiguous().transpose(1, 2)
        embs=m-t
        er = torch.randn([1, m, self.s], device='cuda')
        er=er[:,embs:,:].unsqueeze(0).contiguous()
        qer=torch.matmul(q,er.transpose(-1,-2))
        qer=self._mask_positions(qer)
        bias  = self._skew(qer)
        bias=bias.reshape(b,t,t)
        q= torch.cat(
            [q * torch.sin(weight_index[:, :t, :] / m),
             q * torch.cos(weight_index[:, :t, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :t, :] / m),
             k * torch.cos(weight_index[:, :t, :] / m)],
            dim=-1)
        q=q.reshape(b,c,dk*2)
        k=k.reshape(b,c,dk*2)
        # Calculate the quadratic attention.
        qk = torch.einsum(' bnd,bmd -> bnm ', q, k)

        kernel = torch.square(torch.relu(qk+bias))
        kernel *= mask
        x = u * torch.einsum(' bnm,bme -> bne ', kernel, v)

        x = self.dropout(self.x_proj(x)+shortcut)
        return x, kernel

    def inference(self, inputs, mask, pos, cache=None):
        context, attn_weights = self.forward(inputs, mask, pos)
        return context, attn_weights, cache
