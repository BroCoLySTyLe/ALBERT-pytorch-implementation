import torch
import torch.nn as nn

from Transformer import Transformer


class ALBERT(nn.Module):
    def __init__(self, layer_iter=12, num_group=1, model_hidden=768, num_head=12, dropout=0.1):

        super().__init__()

        """
        embedding not implemented yet 
        """
        
        self.layer_iter = layer_iter
        self.num_group=num_group
        self.model_hidden = model_hidden
        self.num_head = num_head
        self.feed_forward_hidden = model_hidden * 4
        self.transformer_layer_group = nn.ModuleList([Transformer(model_hidden, model_hidden*4, num_head, dropout) for _ in range(num_group)])

    def forward(self, x, mask):
        """
        embedding not implemented yet 
        """
                                                                                              
        for i in range(self.layer_iter):
            
            # group stacking version =>  group_index = i % self.num_group
            
            group_index = (i * self.num_group) // self.layer_iter 
            x = self.transformer_layer_group[group_index](x,mask)

        return x
