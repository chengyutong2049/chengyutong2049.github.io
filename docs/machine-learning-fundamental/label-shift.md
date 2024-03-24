---
layout: default
title: Label Shift
parent: Machine Learning Fundamental
permalink: /docs/machine-learning-fundamental/label-shift
nav_order: 2
math: mathjax
---

# Label Shift

## Problem Formulation

* A common assumption in machine learning is that the training set and test set are drawn from the
same distribution
* However, this assumption often does not hold in practice when models are
deployed in the real world
* One common type of distribution shift is label shift, where the conditional distribution $$p(x\mid y)$$ is fixed but the label distribution $$p(y)$$ changes over time. 

