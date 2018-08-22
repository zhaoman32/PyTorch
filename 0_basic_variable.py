# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable

# torch 中 Variable 模块


tensor = torch.FloatTensor([[1, 2], [3, 4]])
#  requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)
print(t_out)
print(v_out) # v_out = 0.25 * (var * var)

v_out.backward()
# d(v_out)/d(variable) = 1/4 * 2 * var = var/2
print(variable.grad)
print(variable)
print(variable.data)
print(variable.data.numpy())