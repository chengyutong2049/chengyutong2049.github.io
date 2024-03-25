---
layout: default
title: Transformer
parent: Deep Learning Fundamental
permalink: /docs/deep-learning-fundamental/transformer
nav_order: 2
math: mathjax
---

# Transformer
{: .fs-9 }
<!-- {: .no_toc } -->

<!-- [Github repo](https://github.com/chengyutong2049/GRACE) -->
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

##  3 broad categories of transformer models based on their training methodologies

<!-- * GPT-like (auto-regressive) -->
<!-- * BERT-like (auto-encoding) -->
<!-- * BART/T5-like (sequence-to-sequence) -->

### Sequence-to-Sequence Models
* Sequence to Sequence Transformer models adopt a relatively straightforward approach by embedding an entire sequence into a higher dimension, which is then decoded by a decoder.
* These were primarily designed for translation tasks, due to their excellence at mapping sequences between languages.
* `BART/T5`

![](/assets/images/DL/transformer(f1).webp){: width="500"}

### Autoregressive Models
* Autoregressive models leverage the prior tokens to predict the next token iteratively. 
* They employ probabilistic inference to generate text, relying on the decoder component of the transformer.
* `GPT`

### Autoencoding Models
* Autoencoding models are specifically oriented toward language understanding and classification tasks, aiming to capture meaningful representations of input data in their encoded form.
* The training process of autoencoding models often involves bidirectionality, which means they consider both the forward and backward context of the input sequence. 
* `BERT`