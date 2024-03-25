---
layout: default
title: omegaconf
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/omegaconf
math: mathjax
---

# omegaconf
{: .fs-9 }

[Document](https://omegaconf.readthedocs.io/en/2.3_branch/).
{: .fs-6 .fw-300 }


## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
## register_new_resolver
```python
OmegaConf.register_new_resolver("uuid", lambda: 1)
```

* Here, a new resolver named `"uuid"` is registered with OmegaConf, which is a library for managing configurations in Python applications. 
* This resolver, when called, will always return `1`. 
* Resolvers in OmegaConf allow for dynamic values in the configuration files, which are evaluated at runtime. 
* This particular resolver doesn't do much useful as it's defined but serves as an example of how to set up a custom resolver.

