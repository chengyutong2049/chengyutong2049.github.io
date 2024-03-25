# Python 3 program to demonstrate torch.stack() method 
# for two one dimensional tensors 
# importing torch 
import torch 

# creating tensors 
x = torch.tensor([1.,3.,6.,10.]) 
y = torch.tensor([2.,7.,9.,13.]) 

# printing above created tensors 
print("Tensor x:", x) 
print("Tensor y:", y) 

# join above tensor using "torch.stack()" 
print("join tensors:") 
t = torch.stack((x,y)) 

# print final tensor after join 
print(t) 

print("join tensors dimension 0:") 
t = torch.stack((x,y), dim = 0) 
print(t) 

print("join tensors dimension 1:") 
t = torch.stack((x,y), dim = 1) 
print(t) 
