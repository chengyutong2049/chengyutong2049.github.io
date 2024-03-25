---
layout: default
title: logging
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/logging
math: mathjax
---

# logging
{: .fs-9 }

[Document](https://docs.python.org/3/library/logging.html).
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## basicConfig
```python
logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
```

* This line configures the root logger with a specific format for log messages and sets the logging level to `INFO`. 
* This means that all log messages with a level of `INFO` or higher (e.g., `WARNING`, `ERROR`, `CRITICAL`) will be output. 
* The format specified includes the timestamp (`%(asctime)s`), the log level (`%(levelname)s`), the filename and line number where the log call was made (`[%(filename)s:%(lineno)d]`), and the actual log message (`%(message)s`).

## getLogger
```python
LOG = logging.getLogger(__name__)
```
* This line creates a logger with the name of the current module (`__name__`). 
* This is useful for module-specific logging. 
* Since the logger is configured with the name of the module, it's possible to adjust logging levels or handlers for this specific logger separately from the root logger or other loggers in the application.
