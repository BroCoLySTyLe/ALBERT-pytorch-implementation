import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, model_hidden, num_head, dropout=0.1):
        super().__init__()
        
        self.num_head = num_head
        self.head_hidden = model_hidden // num_head
        self.query_linear = torch.nn.Linear(model_hidden, model_hidden)
        self.key_linear = torch.nn.Linear(model_hidden, model_hidden)
        self.value_linear = torch.nn.Linear(model_hidden, model_hidden)
        self.output_linear = torch.nn.Linear(model_hidden, model_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        
        
        assert query.shape == key.shape == value.shape , "Query, Key, Value Shape Error"
        
        
        batch_size = query.shape[0]
        seq_len = query.shape[1] 
        
        query = self.query_linear(query).view(batch_size, seq_len, self.num_head, self.head_hidden).permute(0,2,1,3)
        key = self.key_linear(key).view(batch_size, seq_len, self.num_head, self.head_hidden).permute(0,2,1,3)
        value = self.value_linear(value).view(batch_size, seq_len, self.num_head, self.head_hidden).permute(0,2,1,3)
        
        attn_score = torch.matmul(query, key.permute(0,1,3,2))
        
        if mask is not None:
            assert mask.shape == torch.Size([batch_size, seq_len]) , "Attention mask Shape Error"
            mask_tensor = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1)
            mask_tensor = mask_tensor.type(torch.float)
            mask_tensor = torch.where(mask_tensor==0, torch.tensor(-1e+10, dtype=torch.float), mask_tensor)
            mask_tensor = torch.where(mask_tensor==1, torch.tensor(0, dtype=torch.float), mask_tensor)
            attn_score += mask_tensor
        
        attn_ratio = torch.nn.functional.softmax(attn_score, dim=-1)
        attned_value = torch.matmul(attn_ratio, value)
        attned_value = attned_value.permute(0,2,1,3).reshape(batch_size, seq_len,-1)
        attned_value = self.dropout(attned_value)
        
        return attned_value