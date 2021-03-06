import torch
import torch.nn as nn
import math
from typing import Optional

class ALBERTTokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, token_embedding_size: int = 128, dropout = 0.1):
        super().__init__()
        
        self.token = TokenEmbedding(vocab_size = vocab_size, token_embedding_size = token_embedding_size)
        self.segment = SegmentEmbedding(token_embedding_size = token_embedding_size)
        self.position = PositionalEmbedding(token_embedding_size = token_embedding_size)
        self.token_embedding_size = token_embedding_size

    def forward(self, input_ids: torch.IntTensor, segment_ids: Optional[torch.IntTensor] = None) -> torch.Tensor:
        if segment_ids is not None:
            return self.token(input_ids) + self.position(input_ids) + self.segment(segment_ids) 
        else:
            return self.token(input_ids) + self.position(input_ids)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, token_embedding_size: int = 128):
        super().__init__(vocab_size, token_embedding_size, padding_idx = 0)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, token_embedding_size: int = 128):
        super().__init__(3, token_embedding_size, padding_idx = 0)


## PositionalEmbedding from pytorch PositinalEncoding
class PositionalEmbedding(nn.Module):

    def __init__(self, token_embedding_size: int = 128, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, token_embedding_size, 2) * (-math.log(10000.0) / token_embedding_size))
        pe = torch.zeros(max_len, token_embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        pe.require_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x =  self.pe[:,:x.shape[1]]
        return x