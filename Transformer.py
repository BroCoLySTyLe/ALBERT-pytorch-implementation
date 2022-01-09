import torch
import torch.nn as nn
from typing import Optional

from MultiHeadAttention import MultiHeadAttention
class Transformer(nn.Module):
    def __init__(self, model_hidden : int , feed_forward_hidden : int , num_head : int , dropout : int = 0.1):
        super().__init__()
        
        self.multihead_attention = MultiHeadAttention(model_hidden=model_hidden, num_head=num_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_ff_in = nn.Linear(model_hidden, feed_forward_hidden)
        self.position_wise_ff_out = nn.Linear(feed_forward_hidden, model_hidden)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(model_hidden)      
  
    def forward(self, input_tensor : torch.Tensor, mask : Optional[torch.ByteTensor]) -> torch.Tensor:
        
        attn_out = self.multihead_attention(input_tensor, input_tensor, input_tensor, mask=mask)
        in_sub_layer = input_tensor + self.dropout(attn_out)
        in_layer_norm = self.layer_norm(in_sub_layer)
        position_wise_ff_output = self.position_wise_ff_out(self.gelu(self.position_wise_ff_in(in_layer_norm)))
        out_sub_layer = in_layer_norm + self.dropout(position_wise_ff_output)
        out_layer_norm = self.layer_norm(out_sub_layer)
        
        return self.dropout(out_layer_norm)