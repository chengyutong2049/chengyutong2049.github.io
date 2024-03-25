---
layout: default
title: GPU
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/gpu
math: mathjax
---

# GPU
{: .fs-9 }

<!-- [Document](https://docs.wandb.ai/). -->

<!-- [Account](https://wandb.ai/ci-ci). -->

{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## view Nvidia GPU usage
`nvidia-smi`

```cmd
(base) yutong@seclab-gpu:~$ nvidia-smi
Mon Mar 25 15:56:25 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  | 00000000:01:00.0 Off |                    0 |
| N/A   49C    P0             203W / 500W |  74966MiB / 81920MiB |     95%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-80GB          On  | 00000000:41:00.0 Off |                    0 |
| N/A   62C    P0             327W / 500W |  52448MiB / 81920MiB |     89%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM4-80GB          On  | 00000000:81:00.0 Off |                    0 |
| N/A   51C    P0             337W / 500W |  51050MiB / 81920MiB |     86%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM4-80GB          On  | 00000000:C1:00.0 Off |                    0 |
| N/A   52C    P0             374W / 500W |  40472MiB / 81920MiB |     89%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    767250      C   python                                    53794MiB |
|    0   N/A  N/A    810421      C   python                                      860MiB |
|    1   N/A  N/A    767257      C   python                                    52434MiB |
|    2   N/A  N/A    767263      C   python                                    51036MiB |
|    3   N/A  N/A    767270      C   python                                    40458MiB |
+---------------------------------------------------------------------------------------+```
```

## slurm
```python
os.getenv('SLURM_JOBID')
```

* SLURM, a popular workload manager for clusters.
* If the script is not running in a SLURM environment, os.getenv('SLURM_JOBID') will return None.

