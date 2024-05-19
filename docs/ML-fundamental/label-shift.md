---
layout: default
title: Label Shift
parent: Machine Learning Fundamental
permalink: /docs/machine-learning-fundamental/label-shift
nav_order: 2
math: mathjax
---

# Label Shift

Here we will introduce the fundamental concepts of label shift in machine learning
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Problem Formulation

* A common assumption in machine learning is that the training set and test set are drawn from the
same distribution
* However, this assumption often does not hold in practice when models are
deployed in the real world
* One common type of distribution shift is label shift, where the conditional distribution $$p(x\mid y)$$ is fixed but the label distribution $$p(y)$$ changes over time. 

