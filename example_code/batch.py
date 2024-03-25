import numpy as np
import torch
import torch.utils.data as data_utils

# Create toy data
x = np.linspace(start=1, stop=10, num=10)
x = np.array([np.random.normal(size=len(x)) for i in range(100)])
print(x.shape)
# >> (100,10)

# Create DataLoader
input_as_tensor = torch.from_numpy(x).float()
dataset = data_utils.TensorDataset(input_as_tensor)
dataloader = data_utils.DataLoader(dataset,
                                   batch_size=100,
                                  )
batch = next(iter(dataloader))

print(type(batch))
# >> <class 'list'>

print(len(batch))
# >> 1

print(type(batch[0]))
# >> class 'torch.Tensor'>