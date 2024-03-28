import torch

# Create a tensor of shape (1, N), for example, (1, 5)
tensor_1_N = torch.randn(1, 5)  # Random tensor for demonstration
print("Original tensor shape:", tensor_1_N.shape)
print("Original tensor:", tensor_1_N)

# Convert it to a tensor of shape (N) by squeezing
tensor_N = tensor_1_N.squeeze()
print("Squeezed tensor shape:", tensor_N.shape)
print("Squeezed tensor:", tensor_N)