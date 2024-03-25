---
layout: default
title: hydra
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/hydra
math: mathjax
---

# logging
{: .fs-9 }

[Document](https://hydra.cc/).
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
## Decorating the main function
```python
@hydra.main(config_path="config", config_name="config", version_base="1.2")
```
* This line is a decorator provided by Hydra, a framework for elegantly configuring complex applications. The decorator is applied to the run function (not shown in the snippet you provided, but it's the function immediately following this decorator in your code). Here's what each argument to the decorator specifies:
* config_path="config": This tells Hydra where to look for configuration files. Hydra will search for configuration files in a directory named config relative to the location where the script is run.
* config_name="config": This specifies the name of the default configuration file (without the file extension) that Hydra will use. Hydra expects this file to be in the config_path directory. So, in this case, Hydra will look for a file named config.yaml (or another supported format like .json or .yml) in the config directory.
* version_base="1.2": This sets the version of the Hydra configuration. Hydra versions its configuration syntax and behavior, and specifying the version ensures that the correct parsing and validation logic is applied. Version "1.2" refers to a specific set of features and syntax supported by Hydra.
* When the script is executed, Hydra initializes the application's configuration according to the specified parameters. It loads the configuration from the specified file, potentially overriding it with command line arguments or environment variables, and then passes the resulting configuration object (DictConfig) as an argument to the decorated function (run in your case). This allows your application to be highly configurable, with the ability to easily switch between different settings or environments by simply changing the configuration file or command line arguments.