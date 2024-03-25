---
layout: default
title: checkpoint
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/checkpoint
math: mathjax
---

# Checkpoint
{: .fs-9 }

<!-- [Document](https://pytorch.org/docs/stable/index.html). -->
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
## What is Checkpointing in Machine Learning?
* Checkpointing in machine learning is the technique of preserving intermediate models throughout the training process to resume training from the most recent point in the event of a system breakdown or stoppage. It entails regularly preserving a neural network’s or checkpoint machine learning model’s weights, biases, and other parameters during training, restoring the model to a prior state if training is halted or fails.
* Checkpointing may be done manually by the user or automatically with the help of a framework or library that supports the capability. TensorFlow, PyTorch, and Keras, for example, have built-in model checkpoint capabilities that let users save and restore models during training.

