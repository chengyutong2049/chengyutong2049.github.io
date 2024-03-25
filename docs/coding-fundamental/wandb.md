---
layout: default
title: wandb
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/wandb
math: mathjax
---

# wandb
{: .fs-9 }

[Document](https://docs.wandb.ai/).

[Account](https://wandb.ai/ci-ci).

{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction
Weights & Biases is the machine learning platform for developers to build better models faster. Use W&B's lightweight, interoperable tools to quickly track experiments, version and iterate on datasets, evaluate model performance, reproduce models, visualize results and spot regressions, and share findings with colleagues.

## Running
[Problem of wandb in train.py](https://github.com/ultralytics/yolov5/issues/5772)  
```cmd
wandb: Appending key for api.wandb.ai to your netrc file: /home/yutong/.netrc
wandb: Run data is saved locally in /home/yutong/GRACE/wandb/run-20240325_112919-y3h4mzo2
wandb: ⭐️ View project at https://wandb.ai/ci-ci/grace
wandb: 🚀 View run at https://wandb.ai/ci-ci/grace/runs/y3h4mzo2
``` 


## wandb config
```python
    if config.wandb:
        wandb.init(project=config.wandb_project_name, config=config, mode=config.wandb_mode)
        if not config.wandb_run_name:
            wandb.run.name = f"cici-run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        wandb.run.save()
        wandb.config = config
```
* `wandb.init()` spawns a new background process to log data to a run, and it also syncs data to wandb.ai by default, so you can see live visualizations.
* `wandb.run.save()` saves the run to the database.

