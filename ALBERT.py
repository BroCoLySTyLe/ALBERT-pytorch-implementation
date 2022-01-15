import torch
import torch.nn as nn
from Transformer import Transformer
from Embeddings import ALBERTTokenEmbedding
from typing import Optional

class ALBERT(nn.Module):
    def __init__(self, vocab_size: int, layer_iter: int = 12, num_group: int = 1, token_embedding_size: int = 128,
                 model_hidden: int = 768, num_head: int = 12, dropout: float = 0.1):
        super().__init__()

        assert model_hidden % num_head == 0 , "Model Hidden Size and Number of Attention Head Missmatching"

        self.ALBERTTokenEmbedding = ALBERTTokenEmbedding(vocab_size)
        self.token_to_hidden_projection_layer = torch.nn.Linear(token_embedding_size, model_hidden)
        self.layer_iter = layer_iter
        self.num_group = num_group
        self.model_hidden = model_hidden
        self.num_head = num_head
        self.feed_forward_hidden = model_hidden * 4
        self.transformer_layer_group = nn.ModuleList(
            [Transformer(model_hidden, model_hidden * 4, num_head, dropout) for _ in range(num_group)])

    def forward(self, input_ids: torch.IntTensor, segment_ids: Optional[torch.IntTensor] = None,
                mask: Optional[torch.ByteTensor] = None) -> torch.Tensor:

        if segment_ids is not None:
            assert input_ids.shape == segment_ids.shape, "input_ids and segment_ids shape Missmatching"
            token_embedding = self.ALBERTTokenEmbedding(input_ids, segment_ids)
        else:
            token_embedding = self.ALBERTTokenEmbedding(input_ids)
        input_hidden = self.token_to_hidden_projection_layer(token_embedding)
        for i in range(self.layer_iter):
            # If you want to iterate staked layers, Use group_index = i % self.num_group
            group_index = (i * self.num_group) // self.layer_iter
            #group_index = i % self.num_group
            if i == 0:
                x = self.transformer_layer_group[group_index](input_hidden,mask)
            else:
                x = self.transformer_layer_group[group_index](x,mask)

        return x
