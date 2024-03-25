---
layout: default
title: python
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/python
math: mathjax
---

# Python
{: .fs-9 }

<!-- [Document](https://docs.python.org/3/library/logging.html). -->
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## zip()
```python
for x, y in zip(questions[:1000], answers[:1000]):
            self.data.append({
                "text": x,
                "labels": y
            })
```
* The zip function is used to combine two lists into a single list of tuples.
* In this example, the `questions` and `answers` lists are combined into a single **list of tuples**, which is then used to create a new list of dictionaries. Each dictionary contains a question and its corresponding answer.
