import torch



x = torch.randn(2, 4, 5)
print(x)
y0 = torch.argmax(x,0)
print(y0)
