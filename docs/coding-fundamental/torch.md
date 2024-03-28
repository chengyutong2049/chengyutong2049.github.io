---
layout: default
title: torch
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/torch
math: mathjax
---

# torch
{: .fs-9 }

[Document](https://pytorch.org/docs/stable/index.html).
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
## torch.nn.Module
```python
class QAModel(torch.nn.Module):
    def __init__(self, config):
        super(QAModel, self).__init__()
        self.model = get_hf_model(config).eval()
        self.tokenizer = get_tokenizer(config)
        self.device = config["device"]

    def forward(self, batch):
        logits = []
        self.loss = []
        for item in batch["text"]:
            item = {f"{k1}" : v1.to(self.device) for k1, v1 in item.items()}
            output = self.model(**item)
            logits.append(output.logits)
            try:
                self.loss.append(output.loss)
            except:
                pass
        self.loss = torch.stack(self.loss).mean()
        return torch.stack(logits)

    def get_loss(self, logits, batch):
        return self.loss
```
*  defines a new class named QAModel that inherits from torch.nn.Module

### super
```python
 super(QAModel, self).__init__()
```
* This line calls the constructor of the parent class, `torch.nn.Module`, with the arguments `QAModel` and `self`. These arguments are passed to super() to specify the class (QAModel) and the instance (self) from which the superclass methods are being called.
* However, in Python 3, it's more common to see super() called without arguments inside a class method, which implicitly passes the current class and instance. 


### model.modules()
```python
for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = config.dropout
                n_reset += 1
```
* This line iterates over all modules in the model and checks if the module is an instance of `nn.Dropout`.
* The modules in a PyTorch model can be accessed using the `model.modules()` method, which returns an iterator over all modules in the model, including submodules.
* If the module is an instance of `nn.Dropout`, the dropout probability `m.p` is set to the value specified in the `config` object, and the counter `n_reset` is incremented.

### Definition of modules in pytorch
[Explain.](https://stackoverflow.com/questions/51804692/what-exactly-is-the-definition-of-a-module-in-pytorch)

* Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes. Submodules assigned in this way will be registered, and will have their parameters converted too when you call .cuda(), etc.
* All network components should inherit from nn.Module and override the forward() method. That is about it, as far as the boilerplate is concerned. Inheriting from nn.Module provides functionality to your component. For example, it makes it keep track of its trainable parameters, you can swap it between CPU and GPU with the .to(device) method, where device can be a CPU device torch.device("cpu") or CUDA device torch.device("cuda:0").

### model.eval()
[Explain.](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)

* model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn them off during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation:

* Sets model in evaluation (inference) mode:

  • normalisation layers use running statistics
  • de-activates Dropout layers

![](/assets/images/Code/eval(1).jpg){:"width"=70%}


### batch of data
```python
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
```
* batch is a list containing a single element, which is a torch.Tensor. This is because the DataLoader is configured to return a single batch of data with a batch size of 100. The batch contains 100 samples, each with 10 features. The shape of the tensor is (100, 10), corresponding to the batch size and feature dimensions.
```python
for item in batch["text"]:
            item = {f"{k1}" : v1.to(self.device) for k1, v1 in item.items()}
            output = self.model(**item)
            logits.append(output.logits)
            try:
                self.loss.append(output.loss)
            except:
                pass
```
* In PyTorch, a batch is often represented as a tensor or a list/dictionary of tensors, depending on the complexity of the data and the model's requirements. The line `for item in batch["text"]:` suggests that, in this context, `batch` is expected to be a dictionary where one of the keys is `"text"`, and the value associated with this key is an iterable collection of items (likely tensors) that represent the input data to be processed.

### Batch Normalization
[Explain.](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)

### Layer normalization

### calling forward function
```python
output = self.model(**item)
```
* In PyTorch, when you call a model instance directly with arguments, it internally calls the model's forward method, passing those arguments. 

### logits
```python
output = self.model(**item)
logits.append(output.logits)
```
* `.logits`: This attribute of the output object contains the logits. For a classification model, each logit corresponds to the score of each class before applying the softmax function. For example, in a model trained to classify images into three categories, the logits might be a tensor like [2.0, -1.0, 0.5], indicating the raw scores for each of the three classes.

### torch.stack
```python
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
```
```cmd
Tensor x: tensor([ 1.,  3.,  6., 10.])
Tensor y: tensor([ 2.,  7.,  9., 13.])
join tensors:
tensor([[ 1.,  3.,  6., 10.],
        [ 2.,  7.,  9., 13.]])
join tensors dimension 0:
tensor([[ 1.,  3.,  6., 10.],
        [ 2.,  7.,  9., 13.]])
join tensors dimension 1:
tensor([[ 1.,  2.],
        [ 3.,  7.],
        [ 6.,  9.],
        [10., 13.]])
```

### torch.utils.data.Dataset
![](/assets/images/Code/dataset(1).jpg){:"width"=70%}

[Code.](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset)

### torch.utils.data.DataLoader
[Code.](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

* Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

The DataLoader supports both map-style and iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.
```python
edit_loader = DataLoader(edits, batch_size=1, shuffle=True)
```
* This line creates a DataLoader object named `edit_loader` that loads data from the `edits` dataset with a batch size of 1 and shuffles the data before each epoch. The DataLoader is used to iterate over the dataset in batches during training or evaluation.
* `dataset` (Dataset) – dataset from which to load the data.
* `batch_size` (int, optional) – how many samples per batch to load (default: 1).
  * number of samples we want to pass into the training loop at each iteration
* `shuffle` (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
* `batch_size=1` means that each batch will contain a single sample from the dataset. This is useful when you want to process each sample individually, such as in the case of sequence-to-sequence models or models that require different input sizes for each sample.
  
### batch_size and GPU memory's relation
* Large Batch Sizes: A large batch size improves GPU utilisation by allowing more parallelism. GPUs are designed to handle massive amounts of data simultaneously, and larger batch sizes enable better exploitation of their computational power.
[Explain.](https://www.linkedin.com/pulse/optimising-gpu-utilisation-finding-ideal-batch-size-maximum-bose#:~:text=The%20Relationship%20Between%20Batch%20Size%20and%20GPU%20Utilisation&text=Large%20Batch%20Sizes%3A%20A%20large,exploitation%20of%20their%20computational%20power.)

### Difference btw DataLoader and Dataset
[Explain.](https://mmengine.readthedocs.io/en/latest/tutorials/dataset.html)
* Typically, a dataset defines the quantity, parsing, and pre-processing of the data, while a dataloader iteratively loads data according to settings such as batch_size, shuffle, num_workers, etc. 
* Datasets are encapsulated with dataloaders and they together constitute the data source.


### pad_token_id
* This specific attribute of the tokenizer represents the ID (a numerical value) used to denote padding tokens in the tokenized input.
  

### padding tokens
* Padding tokens are used to ensure that all sequences in a batch have the same length. This is necessary when working with neural networks that require fixed-size inputs, such as recurrent neural networks (RNNs) or transformers. By padding sequences to a common length, you can efficiently process multiple sequences in parallel without having to handle sequences of varying lengths separately.
* model.tokenizer.pad_token_id is used to retrieve the ID of the padding token from the tokenizer associated with a pre-trained model.
  * Padding tokens are used to fill sequences to a uniform length during batch processing.


### named_parameters
* named_parameters(prefix='', recurse=True, remove_duplicate=True)
* Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
  

### requires_grad=False
* If you want to freeze part of your model and train the rest, you can set requires_grad of the parameters you want to freeze to False.
  ```python
  for n, p in self.model.named_parameters():
            p.requires_grad = False
    ```

## Indexing
* PyTorch modules support indexing to access their submodules if they are stored in an ordered container like torch.nn.Sequential, torch.nn.ModuleList, or a custom module that implements the __getitem__ method to allow such access. 

### __getitem__

#### Map-style datasets
A map-style dataset is one that implements the `__getitem__()` and `__len__()` protocols, and represents a map from (possibly non-integral) indices/keys to data samples.

For example, such a dataset, when accessed with **dataset[idx]**, could read the idx-th image and its corresponding label from a folder on the disk.

See Dataset for more details.

#### Iterable-style datasets
An iterable-style dataset is an instance of a subclass of IterableDataset that implements the `__iter__()` protocol, and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.

For example, such a dataset, when called iter(dataset), could return a stream of data reading from a database, a remote server, or even logs generated in real time.

See IterableDataset for more details.

### DenseReluDense
* This is a specific type of layer or component within the Transformer model. The name suggests it is a feed-forward neural network consisting of two dense (fully connected) layers with a ReLU activation function in between. This pattern is common in Transformer models, where it's used to process the data after the self-attention mechanism within each block. It's also known as a position-wise feed-forward network (FFN).