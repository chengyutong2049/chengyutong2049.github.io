---
layout: default
title: Deep Learning Fundamental
nav_order: 3
has_children: true
permalink: /docs/deep-learning-fundamental
math: mathjax
---

# Deep Learning Fundamental
{: .fs-9 }

Here we will introduce the fundamental concepts of deep learning.
{: .fs-6 .fw-300 }

![](/assets/images/DL/weight&bias(1).png)

## Epsilon in Learning Algorithms
* In optimization algorithms, especially those involving gradients (like SGD - Stochastic Gradient Descent), eps might represent 
  * a small value added to denominators to prevent division by zero, 
  * or a threshold for determining convergence. 
* It ensures numerical stability or serves as a stopping criterion.
