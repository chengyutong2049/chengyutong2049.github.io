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

## logging.info
```python
LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
```
* This line logs an informational message using the logger `LOG`.
```cmd
  2 [2024-03-25 15:48:43,457][__main__][INFO] -
  3 tags: null
  4 batch_size: 1
  5 n_iter: 100
  6 max_n_edits: 1000
  7 reinit: true
  8 dropout: 0.0
  9 device: cuda
 10 wandb: true
 11 wandb_mode: online
 12 wandb_project_name: grace
 13 wandb_run_name: null
 14 metric_period: 50
 15 pretrain: false
 16 load_pretrained: true
 17 re_init_model: false
 18 ckpt: false
 19 ckpt_dir: ./ckpts/
 20 editor:
 21   _name: grace
 22   edit_lr: 1.0
 23   n_iter: 100
 24   eps: 1.0
 25   dist_fn: euc
 26   val_init: cold
 27   val_train: sgd
 28   val_reg: None
 29   reg: early_stop
 30   replacement: replace_prompt
 31   eps_expand: coverage
 32   num_pert: 8
 33 experiment:
 34   task: qa
 35   dataset: zsre
 36   cbase: 1.0
 37 model:
 38   name: google/t5-small-ssm-nq
 39   class_name: AutoModelForSeq2SeqLM
 40   tokenizer_class: AutoTokenizer
 41   tokenizer_name: google/t5-small-ssm-nq
 42   inner_params:
 43   - encoder.block[4].layer[1].DenseReluDense.wo.weight
 44   pt: null
 ```
