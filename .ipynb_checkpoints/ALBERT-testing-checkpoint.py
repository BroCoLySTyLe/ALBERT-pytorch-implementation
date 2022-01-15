import torch
from ALBERT import ALBERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = torch.randint(0,29999,(3,10)).to(device)
segment_ids = torch.IntTensor([[1]*10,[1]*5+[2]*5,[2]*10]).to(device)
input_mask = torch.BoolTensor([[0]*10,[0]*5+[1]*5,[1]*10]).to(device)

print (input_ids.shape)
print (segment_ids.shape)
print (input_mask.shape)
print (input_ids)
print (segment_ids)
print (input_mask)

albert_test=ALBERT(vocab_size=30000)
albert_test.to(device)
print(albert_test(input_ids, segment_ids, mask=input_mask))