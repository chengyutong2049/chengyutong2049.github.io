---
layout: default
title: dropout
parent: Machine Learning Fundamental
permalink: /docs/machine-learning-fundamental/dropout
nav_order: 2
math: mathjax
---

# dropout
{: .no_toc }

Here we will introduce the fundamental concepts of [dropout](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) in machine learning
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Problem Formulation

* The term “dropout” refers to dropping out the nodes (input and hidden layer) in a neural network. All the forward and backwards connections with a dropped node are temporarily removed, thus creating a new network architecture out of the parent network. The nodes are dropped by a dropout probability of p.
* 
![](/assets/images/ML/dropout(1).webp){:width="70%"}

