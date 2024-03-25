---
layout: default
title: CUDA
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/CUDA
math: mathjax
---

# CUDA
{: .fs-9 }

Here we will introduce the fundamental concepts of CUDA.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Compute Unified Device Architecture (CUDA) 

* Nvidia推出的只能用于自家GPU的并行计算框架。只有安装这个框架才能够进行复杂的并行计算。主流的深度学习框架也都是基于CUDA进行GPU并行加速的，几乎无一例外。还有一个叫做cudnn，是针对深度卷积神经网络的加速库。

---
## torch.cuda.is_available
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

* This line checks if a CUDA-capable GPU is available for PyTorch to use (`torch.cuda.is_available()`). If a GPU is available, it sets `device` to use the first GPU (`"cuda:0"`). If not, it falls back to using the CPU (`"cpu"`). This `device` variable is later used to move tensors or models to the appropriate compute device.